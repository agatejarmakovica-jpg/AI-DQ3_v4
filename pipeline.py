#!/usr/bin/env python3
"""
AI-DQ3: semantic variable-aware data quality assessment pipeline
================================================================

Purpose
-------
Develop and evaluate AI-DQ3, a *semantic variable-aware* assessment pipeline
that uses inferred variable meaning to guide Completeness (C), Accuracy (A) and
Reuse-readiness (R) evaluation of tabular healthcare datasets.

What this pipeline IS
---------------------
* A semantic interpretation layer that classifies each variable into a role and
  uses that role to SELECT which quality checks are applied (RQ1).
* A dimension-specific assessment producing an interpretable, decomposable
  C/A/R profile per dataset (RQ3).
* A side-by-side comparison of semantic vs. uniform data-type profiling and its
  effect on issue interpretation and prioritisation (RQ2).
* A human-in-the-loop (HITL) triage layer that ranks candidate issues for expert
  review. HITL is cross-cutting and is NOT a scored dimension.

What this pipeline is NOT
-------------------------
* It does NOT introduce a new anomaly-detection, imputation, or automatic
  data-repair algorithm. Standard methods (IQR, modified z-score, Isolation
  Forest, Local Outlier Factor, TF-IDF cosine, k-anonymity) are reused as-is.
* Median fill is applied ONLY to the matrix fed to IsolationForest / LOF (which
  reject NaN) and its footprint is reported transparently; it never alters the
  dataset that the pipeline returns or the completeness assessment.

Reproducibility rule
---------------------
No dataset-specific schema, clinical range, allowed-value list, target field,
required field, quasi-identifier, or reuse-readiness rule is hardcoded. Such
knowledge is supplied through metadata JSON; generic thresholds/weights live in
config.yaml.

Research questions answered by the emitted artifacts
----------------------------------------------------
RQ1 -> rq1_check_selection_map.csv      (role -> selected checks)
RQ2 -> rq2_semantic_vs_uniform.csv      (semantic vs uniform, all dimensions)
RQ3 -> rq3_quality_profile.csv/.json    (interpretable C/A/R profile)
       quality_profile.md               (manuscript-oriented narrative)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def read_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clip01(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den not in (0, None) else 0.0


def normalize_name(value: Any) -> str:
    return str(value).strip().lower()


def matches_any(text: str, patterns: Sequence[str]) -> bool:
    s = normalize_name(text)
    return any(re.search(pattern, s) for pattern in patterns)


def score_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return clip01(float(value))
    except Exception:
        return 0.0


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# -----------------------------------------------------------------------------
# Input and metadata loading
# -----------------------------------------------------------------------------


def load_dataset(path: Path) -> Tuple[pd.DataFrame, str]:
    """Return (dataframe, encoding_note). encoding_note records non-UTF-8 fallback
    so it can feed the machine-readability facet.

    Delimited text (.csv/.tsv) is read robustly and dataset-agnostically:
      * encoding is auto-detected (UTF-8, else Latin-1 fallback);
      * the field delimiter is chosen from {",", ";", tab, "|"} as the one that
        yields the widest, most row-consistent table (handles European ";"-CSV);
      * a European decimal comma is detected from the raw text (digit-comma-digit
        pattern when the delimiter is not a comma) so numeric columns parse as
        numbers rather than strings.
    No value is hard-coded to any particular file; the choice is made from the
    data itself, so a comma-CSV, a ";"-CSV and a tab file all load correctly."""
    import io as _io

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
            non_utf8 = False
        except UnicodeDecodeError:
            text = raw.decode("latin1")
            non_utf8 = True

        # --- choose delimiter from the data, not from the extension ---
        sample_lines = [ln for ln in text.splitlines() if ln.strip()][:50]
        sample = "\n".join(sample_lines)
        candidates = ["\t"] if suffix == ".tsv" else [",", ";", "\t", "|"]
        best_sep, best_ncols, best_consistency = ",", 0, -1.0
        for sep in candidates:
            try:
                probe = pd.read_csv(_io.StringIO(sample), sep=sep, engine="python")
            except Exception:
                continue
            ncols = probe.shape[1]
            if ncols <= 1:
                continue
            # prefer more columns; break ties by how uniform the row width is
            widths = [len(ln.split(sep)) for ln in sample_lines]
            consistency = (widths.count(ncols) / len(widths)) if widths else 0.0
            if (ncols > best_ncols) or (ncols == best_ncols and consistency > best_consistency):
                best_sep, best_ncols, best_consistency = sep, ncols, consistency

        # --- detect European decimal comma (only meaningful when sep is not ",") ---
        decimal = "."
        if best_sep != "," and re.search(r"\d,\d", text):
            decimal = ","

        df = pd.read_csv(_io.StringIO(text), sep=best_sep, decimal=decimal, engine="python")

        if non_utf8:
            note = "non_utf8_latin1_fallback"
        elif best_sep == ";":
            note = "utf8_semicolon"
        else:
            note = "utf8"
        return df, note

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path), "excel"
    raise ValueError(f"Unsupported data file: {path}")


def normalize_technical_missing(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    markers = {normalize_name(x) for x in config["missingness"].get("technical_missing_markers", [])}
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda x: np.nan if isinstance(x, str) and normalize_name(x) in markers else x
            )
    return out


def find_data_files(data_dir: Path, config: Dict[str, Any]) -> List[Path]:
    extensions = set(config["input"].get("accepted_data_extensions", [".csv", ".xlsx", ".xls"]))
    return sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions])


def default_metadata() -> Dict[str, Any]:
    return {
        "dataset_type": "unspecified_tabular_dataset",
        "id_fields": [],
        "target_fields": [],
        "required_fields": [],
        "critical_fields": [],
        "semantic_roles": {},
        "allowed_values": {},
        "valid_ranges": {},
        "consistency_rules": [],
        "direct_identifiers": [],
        "quasi_identifiers": [],
        "reuse_readiness": {},
        "variable_descriptions": {},
        "documentation_text": "",
    }


def merge_metadata_defaults(metadata: Dict[str, Any]) -> Dict[str, Any]:
    merged = default_metadata()
    merged.update(metadata or {})
    for key, default in default_metadata().items():
        if merged.get(key) is None:
            merged[key] = default
    return merged


def _extract_text_from_file(path: Path) -> str:
    """Extract plain text from a documentation file (.txt/.md/.csv-as-text/.pdf/.docx).
    PDF/DOCX use optional libraries; if unavailable, returns '' with a console note."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".csv", ".tsv"}:
            # A codebook / data dictionary supplied as a delimited file is read as
            # plain text so its variable definitions feed the documentation facet.
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="latin1", errors="ignore")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                except Exception:
                    print(f"AI-DQ3: '{path.name}' is a PDF but no PDF reader is installed "
                          f"(pip install pypdf) — skipping its text.")
                    return ""
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix == ".docx":
            try:
                import docx  # type: ignore
            except Exception:
                print(f"AI-DQ3: '{path.name}' is a DOCX but python-docx is not installed "
                      f"(pip install python-docx) — skipping its text.")
                return ""
            document = docx.Document(str(path))
            return "\n".join(p.text for p in document.paragraphs)
    except Exception as exc:
        print(f"AI-DQ3: could not read documentation file '{path.name}' ({type(exc).__name__}).")
    return ""


def _tokenize_stem(name: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) > 1}


# Filenames that are documentation for the dataset regardless of exact stem match.
_DOC_NAME_HINTS = ("readme", "codebook", "code_book", "code book", "dictionary",
                   "data_dictionary", "datadictionary", "description", "documentation", "notes")

# Files that ship WITH the AI-DQ3 tool itself (scaffolding / instructions / templates)
# and must NEVER be treated as documentation or metadata for a user's dataset.
# Without this guard, a single-dataset run (n_datasets == 1, "attach loose files")
# would sweep in the tool's own guide/template and award a spurious documentation
# score even when the dataset has no accompanying documentation at all.
# Restricted to the tool's exact shipped filenames so a user's own readme/codebook
# (e.g. an uploaded "ReadMe.pdf") is never excluded.
_TOOL_SCAFFOLDING = {"metadata_guide.md", "metadata_template.json"}


def find_related_metadata(dataset_path: Path, metadata_dir: Path, config: Dict[str, Any],
                          n_datasets: int = 1) -> Dict[str, Any]:
    """Attach metadata to a dataset. Structured metadata comes from
    '<stem>_metadata.json'; documentation (readme / codebook / data dictionary,
    including PDF and DOCX) is matched by shared filename tokens, by documentation
    name hints, or — when there is a single dataset — attached regardless. The
    documentation text feeds the reuse-readiness facets, so an uploaded codebook
    legitimately raises the documentation score."""
    if not metadata_dir.exists():
        return default_metadata()

    accepted = set(config["input"].get("accepted_metadata_extensions", [".json", ".txt", ".md"]))
    accepted |= {".pdf", ".docx", ".csv", ".tsv"}
    ds_tokens = _tokenize_stem(dataset_path.stem)
    metadata: Dict[str, Any] = {}
    documentation_chunks: List[str] = []

    for path in sorted(metadata_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in accepted:
            continue
        if path.name.lower() in _TOOL_SCAFFOLDING:
            continue  # never treat the tool's own guide/template as dataset documentation
        raw_stem = path.stem.lower()
        meta_tokens = _tokenize_stem(raw_stem.replace("metadata", "").replace("code book", "")
                                     .replace("codebook", "").replace("readme", "")
                                     .replace("dictionary", ""))
        shared = ds_tokens & meta_tokens
        is_doc_named = any(h in raw_stem for h in _DOC_NAME_HINTS)
        # Relate if: tokens overlap, OR it's a documentation-named file, OR there
        # is only one dataset (so loose files clearly belong to it).
        is_related = bool(shared) or is_doc_named or n_datasets == 1
        if not is_related:
            continue

        if path.suffix.lower() == ".json":
            # Only merge structured metadata when it clearly matches this dataset.
            if shared or n_datasets == 1:
                try:
                    with path.open("r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        loaded.pop("_instructions", None)
                        metadata.update(loaded)
                except Exception as exc:
                    print(f"AI-DQ3: could not parse metadata JSON '{path.name}' ({type(exc).__name__}).")
        else:
            text = _extract_text_from_file(path)
            if text.strip():
                documentation_chunks.append(f"[{path.name}]\n{text}")

    if documentation_chunks:
        existing = metadata.get("documentation_text", "")
        metadata["documentation_text"] = (existing + "\n\n" + "\n\n".join(documentation_chunks)).strip()
    return merge_metadata_defaults(metadata)


def auto_infer_metadata(df: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Zero-config inference: detect identifiers, direct/quasi identifiers and a
    likely target from the data alone, so ANY tabular dataset can be assessed
    without a metadata file. User-supplied values always win; auto-inference only
    fills gaps. Returns a record of what was auto-detected (for transparency)."""
    auto_cfg = config.get("auto_inference", {})
    detected: Dict[str, Any] = {"id_fields": [], "target_fields": [], "direct_identifiers": [],
                                "quasi_identifiers": [], "method": "auto" if auto_cfg.get("enabled", True) else "disabled"}
    if not auto_cfg.get("enabled", True):
        return detected

    n = max(len(df), 1)
    role_cfg = config["semantic_role_inference"]
    high_uniq = role_cfg.get("high_uniqueness_threshold", 0.85)
    qi_min = auto_cfg.get("quasi_identifier_cardinality_min", 2)
    qi_max_ratio = auto_cfg.get("quasi_identifier_cardinality_max_ratio", 0.50)
    max_qi = auto_cfg.get("max_auto_quasi_identifiers", 8)
    direct_pat = auto_cfg.get("direct_identifier_patterns", [])
    quasi_pat = auto_cfg.get("quasi_identifier_patterns", [])

    user_ids = {normalize_name(x) for x in metadata.get("id_fields", [])}
    user_targets = {normalize_name(x) for x in metadata.get("target_fields", [])}
    user_direct = {normalize_name(x) for x in metadata.get("direct_identifiers", [])}
    user_quasi = {normalize_name(x) for x in metadata.get("quasi_identifiers", [])}

    for col in df.columns:
        col_l = normalize_name(col)
        non_missing = df[col].dropna()
        if len(non_missing) == 0:
            continue
        uniqueness = non_missing.nunique() / n
        numeric_rate = pd.to_numeric(non_missing, errors="coerce").notna().mean()

        # Identifier: must be near-unique AND key-like (integer codes / string
        # keys), so continuous measurements that happen to be near-unique are not
        # mistaken for identifiers.
        if col_l not in user_ids and _is_key_like(non_missing, config):
            if (matches_any(col, role_cfg.get("id_patterns", [])) and uniqueness >= high_uniq) or uniqueness >= 0.98:
                detected["id_fields"].append(col)

        # Direct identifier: name pattern match.
        if col_l not in user_direct and matches_any(col, direct_pat):
            detected["direct_identifiers"].append(col)

        # Target: name pattern match (kept conservative; only by name).
        if col_l not in user_targets and matches_any(col, role_cfg.get("target_patterns", [])):
            detected["target_fields"].append(col)

    # Quasi-identifiers: demographic/geo/temporal name patterns, OR low-to-medium
    # cardinality columns that are not identifiers/targets/direct-identifiers/free-text.
    excluded = (set(detected["id_fields"]) | set(detected["target_fields"]) | set(detected["direct_identifiers"])
                | set(metadata.get("id_fields", [])) | set(metadata.get("target_fields", []))
                | set(metadata.get("direct_identifiers", [])))
    excluded_l = {normalize_name(x) for x in excluded}
    for col in df.columns:
        col_l = normalize_name(col)
        if col_l in excluded_l or col_l in user_quasi:
            continue
        non_missing = df[col].dropna()
        if len(non_missing) == 0:
            continue
        uniqueness = non_missing.nunique() / n
        cardinality = non_missing.nunique()
        by_name = matches_any(col, quasi_pat)
        by_distribution = (qi_min <= cardinality and uniqueness <= qi_max_ratio
                           and not matches_any(col, role_cfg.get("text_patterns", [])))
        if by_name or by_distribution:
            detected["quasi_identifiers"].append(col)
    # Cap auto quasi-identifiers; prefer name-matched ones first.
    if len(detected["quasi_identifiers"]) > max_qi:
        named = [c for c in detected["quasi_identifiers"] if matches_any(c, quasi_pat)]
        rest = [c for c in detected["quasi_identifiers"] if c not in named]
        detected["quasi_identifiers"] = (named + rest)[:max_qi]

    return detected


def apply_auto_inference(metadata: Dict[str, Any], detected: Dict[str, Any]) -> Dict[str, Any]:
    """Merge auto-detected fields UNDER user metadata (user always wins)."""
    merged = dict(metadata)
    for key in ["id_fields", "target_fields", "direct_identifiers", "quasi_identifiers"]:
        user_vals = list(metadata.get(key, []) or [])
        auto_vals = [c for c in detected.get(key, []) if c not in user_vals]
        merged[key] = user_vals + auto_vals
    merged["_auto_inferred"] = detected
    return merged


# -----------------------------------------------------------------------------
# Semantic and uniform profiling  (RQ1 / RQ2 foundation)
# -----------------------------------------------------------------------------


def is_binary_series(series: pd.Series) -> bool:
    values = {normalize_name(v) for v in series.dropna().unique().tolist()}
    if not values:
        return False
    binary_sets = [{"0", "1"}, {"true", "false"}, {"yes", "no"}, {"y", "n"}, {"male", "female"}, {"m", "f"}]
    return any(values.issubset(options) and len(values) <= 2 for options in binary_sets)


def _is_key_like(non_missing: pd.Series, config: Dict[str, Any]) -> bool:
    """True when near-unique values look like a record KEY (integer codes or
    string identifiers), False when they are a continuous measurement that merely
    happens to be near-unique. This prevents high-resolution measurements (e.g. a
    lung-function percentage with almost-unique floats) from being mislabelled as
    identifiers. Purely structural, no column-name or domain assumptions."""
    if len(non_missing) == 0:
        return False
    numeric = pd.to_numeric(non_missing.astype(str).str.strip().str.replace(",", ".", regex=False),
                            errors="coerce")
    numeric_rate = numeric.notna().mean()
    if numeric_rate < 0.98:
        return True  # mostly non-numeric near-unique strings -> string keys
    vals = numeric.dropna()
    # Integer-valued (codes/serials) -> key-like; non-integer (continuous) -> not.
    is_integer_valued = bool(np.allclose(vals, vals.round())) and (vals.round().nunique() == vals.nunique())
    return is_integer_valued


def infer_uniform_type(series: pd.Series, config: Dict[str, Any]) -> Tuple[str, float, str]:
    """Baseline profiler: data-type only, no variable-meaning awareness."""
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return "empty_or_missing_only", 0.40, "no non-missing values"

    numeric_rate = pd.to_numeric(non_missing, errors="coerce").notna().mean()
    if numeric_rate >= config["semantic_role_inference"].get("numeric_parse_threshold", 0.98):
        return "numeric", 0.90, "uniform dtype/numeric parsing"

    sample_text = " ".join(non_missing.astype(str).head(20).tolist())
    looks_date_like = bool(re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", sample_text))
    if looks_date_like and pd.to_datetime(non_missing, errors="coerce").notna().mean() >= 0.80:
        return "date_time", 0.80, "uniform date parsing"

    unique_count = non_missing.nunique(dropna=True)
    if is_binary_series(series):
        return "binary", 0.90, "uniform binary values"
    if unique_count <= config["semantic_role_inference"].get("low_cardinality_threshold", 30):
        return "categorical", 0.80, "uniform low-cardinality values"
    return "text_or_high_cardinality", 0.60, "uniform fallback"


# Optional embedding model for semantic role inference. Loaded lazily and cached;
# stays None (and the pipeline falls back to heuristics) if unavailable.
_EMBED_STATE: Dict[str, Any] = {"tried": False, "model": None, "name": None, "proto_vecs": None, "proto_roles": None}


def _get_embedder(config: Dict[str, Any]):
    """Lazily load sentence-transformers + encode role prototypes once. Returns
    None on any failure (missing library, offline, model error) so callers fall
    back to the lexical/distributional heuristics."""
    emb_cfg = config["semantic_role_inference"].get("embedding", {})
    if not emb_cfg.get("enabled", False):
        return None
    if _EMBED_STATE["tried"]:
        return _EMBED_STATE["model"]
    _EMBED_STATE["tried"] = True
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model_name = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        prototypes = emb_cfg.get("role_prototypes", {}) or {}
        roles = list(prototypes.keys())
        proto_vecs = model.encode([prototypes[r] for r in roles], normalize_embeddings=True)
        _EMBED_STATE.update({"model": model, "name": model_name,
                             "proto_vecs": proto_vecs, "proto_roles": roles})
        print(f"AI-DQ3: embedding semantic inference enabled (model: {model_name}).")
        return model
    except Exception as exc:
        print(f"AI-DQ3: embedding semantic inference unavailable ({type(exc).__name__}); "
              f"using lexical/distributional heuristics. Install with "
              f"'pip install -r requirements-semantic.txt' to enable it.")
        _EMBED_STATE["model"] = None
        return None


def embedding_role(col: str, series: pd.Series, metadata: Dict[str, Any], config: Dict[str, Any]):
    """Return (role, confidence, note) from embedding similarity, or None to defer
    to the heuristics. Deterministic given a fixed model."""
    model = _get_embedder(config)
    if model is None:
        return None
    emb_cfg = config["semantic_role_inference"]["embedding"]
    try:
        import numpy as _np
        parts = [str(col).replace("_", " ")]
        desc = (metadata.get("variable_descriptions", {}) or {}).get(col)
        if desc:
            parts.append(str(desc))
        if emb_cfg.get("use_sample_values", True):
            n = int(emb_cfg.get("n_sample_values", 8))
            sample = series.dropna().astype(str).head(n).tolist()
            if sample:
                parts.append("example values: " + ", ".join(sample))
        text = ". ".join(parts)
        vec = model.encode([text], normalize_embeddings=True)[0]
        sims = _np.asarray(_EMBED_STATE["proto_vecs"]) @ _np.asarray(vec)
        best = int(sims.argmax())
        best_sim = float(sims[best])
        if best_sim < float(emb_cfg.get("min_similarity", 0.30)):
            return None
        role = _EMBED_STATE["proto_roles"][best]
        confidence = clip01(0.55 + 0.45 * best_sim)
        return role, confidence, f"embedding match (sim={best_sim:.2f}, model={_EMBED_STATE['name']})"
    except Exception:
        return None


def role_provenance(col: str, series: pd.Series, dtype: str, metadata: Dict[str, Any],
                    config: Dict[str, Any]) -> Dict[str, Any]:
    """Improvement 1 — produce the dual rule-based / embedding role view for a
    column, without changing the existing role used downstream. Returns the six
    required fields. The embedding text combines column name, dtype, sample
    values and (if available) the metadata description, matched against the
    predefined role prototypes by cosine similarity. Falls back to rule-based when
    sentence-transformers is unavailable."""
    rule_role, rule_conf, _ = infer_semantic_role(col, series, metadata, config)
    emb_cfg = config["semantic_role_inference"].get("embedding", {})
    review_thr = config["semantic_role_inference"].get("low_confidence_threshold", 0.65)

    emb_role = None
    emb_conf = None
    model = _get_embedder(config)
    if model is not None:
        try:
            import numpy as _np
            parts = [str(col).replace("_", " "), f"type {dtype}"]
            desc = (metadata.get("variable_descriptions", {}) or {}).get(col)
            if desc:
                parts.append(str(desc))
            if emb_cfg.get("use_sample_values", True):
                n = int(emb_cfg.get("n_sample_values", 8))
                sample = series.dropna().astype(str).head(n).tolist()
                if sample:
                    parts.append("example values: " + ", ".join(sample))
            vec = model.encode([". ".join(parts)], normalize_embeddings=True)[0]
            sims = _np.asarray(_EMBED_STATE["proto_vecs"]) @ _np.asarray(vec)
            best = int(sims.argmax())
            best_sim = float(sims[best])
            if best_sim >= float(emb_cfg.get("min_similarity", 0.30)):
                emb_role = _EMBED_STATE["proto_roles"][best]
                emb_conf = clip01(0.55 + 0.45 * best_sim)
        except Exception:
            emb_role, emb_conf = None, None

    # Final role: keep the existing pipeline behaviour (rule-based unless the
    # embedding layer is enabled AND already integrated by infer_semantic_role).
    final_role, final_conf, _ = infer_semantic_role(col, series, metadata, config)
    if emb_role is not None and emb_role == final_role:
        source = "agreement"
    elif emb_role is not None and final_role == rule_role and emb_role != rule_role:
        source = "rule_based"   # rule kept; embedding differs -> flag for review
    elif emb_role is not None:
        source = "embedding"
    else:
        source = "rule_based"

    disagree = (emb_role is not None and emb_role != rule_role)
    requires_review = bool(final_conf < review_thr or disagree)
    return {
        "semantic_role_rule_based": rule_role,
        "semantic_role_embedding": emb_role if emb_role is not None else "",
        "semantic_role_final": final_role,
        "semantic_role_confidence": round(float(final_conf), 3),
        "semantic_role_source": source,
        "requires_semantic_review": requires_review,
    }


def infer_semantic_role(col: str, series: pd.Series, metadata: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, float, str]:
    """Semantic profiler: variable meaning, driven by metadata overrides first,
    then (optionally) embedding similarity, then generic name/value heuristics.
    Confidence is reported so uncertainty is explicit."""
    overrides = metadata.get("semantic_roles", {}) or {}
    if col in overrides:
        return str(overrides[col]), 1.0, "metadata semantic role override"

    role_cfg = config["semantic_role_inference"]
    col_l = normalize_name(col)
    non_missing = series.dropna()
    missing_rate = safe_div(series.isna().sum(), len(series))

    id_fields = {normalize_name(x) for x in metadata.get("id_fields", [])}
    target_fields = {normalize_name(x) for x in metadata.get("target_fields", [])}
    if col_l in id_fields:
        return "identifier", 1.0, "metadata id field"
    if col_l in target_fields:
        return "target_or_outcome", 1.0, "metadata target field"
    if len(non_missing) == 0:
        return "empty_or_missing_only", 0.40, "no non-missing values"

    # Hard structural signals take precedence over fuzzy semantics.
    unique_rate = safe_div(non_missing.nunique(dropna=True), len(series))
    # A genuine identifier is a KEY: near-unique AND non-continuous (integer codes
    # or string keys), not a high-resolution continuous measurement that merely
    # happens to be near-unique (e.g. a lung-function % with 68/69 distinct floats).
    looks_key_like = _is_key_like(non_missing, config)
    if (matches_any(col, role_cfg.get("id_patterns", [])) and unique_rate >= role_cfg.get("high_uniqueness_threshold", 0.85)
            and looks_key_like):
        return "identifier", 0.90, "configured id-like pattern, high uniqueness, key-like values"
    if is_binary_series(series):
        return "binary", 0.92, "binary observed value set"

    # Optional embedding-based meaning match (between structural and lexical layers).
    emb = embedding_role(col, series, metadata, config)
    if emb is not None:
        emb_role, emb_conf, emb_note = emb
        # Don't let embeddings override clear numeric structure into a text role.
        numeric_rate_quick = numeric_series(series, config).notna().mean()
        if not (numeric_rate_quick >= role_cfg.get("numeric_parse_threshold", 0.98)
                and emb_role in {"free_text", "categorical_nominal", "date_time"}):
            return emb_role, emb_conf, emb_note

    if matches_any(col, role_cfg.get("target_patterns", [])):
        return "target_or_outcome", 0.85, "configured target-like pattern"
    if matches_any(col, role_cfg.get("text_patterns", [])):
        return "free_text", 0.80, "configured text-like pattern"

    # Sentinel-aware numeric fraction: placeholder/exclusion codes are treated as
    # missing, not as evidence that the column is non-numeric. This stops a
    # measurement column with a few exclusion codes from being mis-typed as
    # categorical / mixed / identifier.
    parsed = numeric_series(series, config)              # sentinels & markers -> NaN
    sentinels = detect_sentinel_tokens(series, config)
    present = series.fillna("").astype(str).str.strip()
    sem_markers = {normalize_name(x) for x in config["missingness"].get("semantic_missing_markers", [])}
    usable_mask = ~present.str.lower().isin(sentinels | sem_markers) & series.notna()
    n_usable = int(usable_mask.sum())
    # numeric_rate = fraction of genuinely-present (non-missing, non-sentinel)
    # values that parse as numbers.
    numeric_rate = safe_div(int((parsed.notna() & usable_mask).sum()), n_usable) if n_usable else 0.0
    unique_count = int(parsed.dropna().nunique()) if numeric_rate > 0 else non_missing.nunique(dropna=True)
    measurement_like = matches_any(col, role_cfg.get("measurement_patterns", [])) or col in metadata.get("valid_ranges", {})

    if numeric_rate >= role_cfg.get("numeric_parse_threshold", 0.98):
        if unique_count <= role_cfg.get("low_cardinality_threshold", 30) and not measurement_like:
            return "ordinal_or_categorical", 0.82, f"few numeric-coded values; missing={missing_rate:.2f}"
        if measurement_like and missing_rate >= 0.20:
            return "continuous_measurement_high_missingness", 0.88, f"measurement-like; missing={missing_rate:.2f}"
        if measurement_like:
            return "continuous_measurement", 0.88, "measurement-like numeric variable"
        return "continuous_numeric", 0.80, "numeric variable without domain override"

    if matches_any(col, role_cfg.get("date_patterns", [])):
        if pd.to_datetime(non_missing, errors="coerce").notna().mean() >= 0.60:
            return "date_time", 0.82, "configured date-like pattern"
    if numeric_rate >= role_cfg.get("mixed_numeric_threshold", 0.50):
        return "mixed_numeric_requires_review", 0.55, f"partial numeric parsing={numeric_rate:.2f}"
    if unique_count <= role_cfg.get("low_cardinality_threshold", 30):
        return "categorical_nominal", 0.78, "limited observed categories"
    return "free_text_or_high_cardinality", 0.55, "semantic role requires review"


def build_profiles(df: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    semantic_rows, uniform_rows = [], []
    for col in df.columns:
        uniform_role, uniform_conf, uniform_note = infer_uniform_type(df[col], config)
        semantic_role, semantic_conf, semantic_note = infer_semantic_role(col, df[col], metadata, config)
        base = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": safe_div(df[col].isna().sum(), len(df)),
            "unique_count": int(df[col].nunique(dropna=True)),
            "unique_rate": safe_div(df[col].nunique(dropna=True), len(df)),
            "non_missing_numeric_rate": (
                pd.to_numeric(df[col].dropna(), errors="coerce").notna().mean() if len(df[col].dropna()) else 0.0
            ),
            "sample_values": ", ".join([str(x) for x in df[col].dropna().head(5).tolist()]),
        }
        semantic_rows.append({**base, "role": semantic_role, "role_confidence": semantic_conf, "notes": semantic_note,
                               **role_provenance(col, df[col], str(df[col].dtype), metadata, config)})
        uniform_rows.append({**base, "role": uniform_role, "role_confidence": uniform_conf, "notes": uniform_note})

    semantic = pd.DataFrame(semantic_rows)
    uniform = pd.DataFrame(uniform_rows)
    comparison = semantic[["column", "role", "role_confidence", "notes"]].merge(
        uniform[["column", "role", "role_confidence", "notes"]], on="column", suffixes=("_semantic", "_uniform")
    )
    comparison["role_changed_by_semantic_layer"] = comparison["role_semantic"] != comparison["role_uniform"]
    comparison["interpretation"] = np.where(
        comparison["role_changed_by_semantic_layer"],
        "Semantic layer changes how checks are selected or interpreted.",
        "Uniform and semantic layers agree.",
    )
    return semantic, uniform, comparison


# -----------------------------------------------------------------------------
# RQ1: explicit role -> check-selection map
# -----------------------------------------------------------------------------

# Roles eligible for distributional anomaly detection (Accuracy dimension).
ANOMALY_ELIGIBLE_ROLES = {
    "continuous_measurement",
    "continuous_measurement_high_missingness",
    "continuous_numeric",
    "ordinal_or_categorical",
}


def derive_check_selection(semantic_profile: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    """Operationalises RQ1: for each variable, record which checks the semantic
    role selects, and contrast with what a uniform numeric rule would do."""
    required = {normalize_name(c) for c in metadata.get("required_fields", [])}
    critical = {normalize_name(c) for c in metadata.get("critical_fields", [])}
    valid_ranges = metadata.get("valid_ranges", {}) or {}
    allowed_values = metadata.get("allowed_values", {}) or {}
    quasi = {normalize_name(c) for c in metadata.get("quasi_identifiers", [])}
    direct = {normalize_name(c) for c in metadata.get("direct_identifiers", [])}
    low_conf = config["semantic_role_inference"].get("low_confidence_threshold", 0.65)

    rows = []
    for _, r in semantic_profile.iterrows():
        col, role, conf = r["column"], r["role"], float(r["role_confidence"])
        col_l = normalize_name(col)
        numeric_rate = float(r["non_missing_numeric_rate"])
        rows.append({
            "column": col,
            "semantic_role": role,
            "role_confidence": conf,
            # Completeness checks
            "completeness_missing_check": True,
            "completeness_required_field": col_l in required,
            "completeness_critical_field": col_l in critical,
            # Accuracy checks
            "accuracy_identifier_uniqueness": role == "identifier",
            "accuracy_type_consistency": ("numeric" in role or "measurement" in role),
            "accuracy_domain_range": col in valid_ranges,
            "accuracy_allowed_values": col in allowed_values,
            "accuracy_anomaly_detection_semantic": role in ANOMALY_ELIGIBLE_ROLES,
            "accuracy_anomaly_detection_uniform": numeric_rate >= 0.98,
            "anomaly_selection_differs": (role in ANOMALY_ELIGIBLE_ROLES) != (numeric_rate >= 0.98),
            # Reuse readiness signals
            "reuse_quasi_identifier": col_l in quasi,
            "reuse_direct_identifier": col_l in direct,
            # Triage signal
            "flag_low_confidence_for_hitl": conf < low_conf,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Issue record helper (shared by all dimensions; feeds HITL triage)
# -----------------------------------------------------------------------------


def issue(dimension: str, method: str, issue_type: str, column: Optional[str], count: int,
          severity: str, hitl_decision: str, rule: str = "") -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "method": method,
        "issue_type": issue_type,
        "column": column,
        "count": int(count),
        "severity": severity,
        "hitl_decision": hitl_decision,
        "rule": rule,
    }


# -----------------------------------------------------------------------------
# COMPLETENESS (C): technical + semantic missingness, role-aware
# -----------------------------------------------------------------------------


def detect_sentinel_tokens(series: pd.Series, config: Dict[str, Any]) -> set:
    """Data-driven (no wordlist): in a predominantly-numeric column, identify the
    small set of repeated NON-numeric string tokens that behave as placeholder /
    exclusion codes rather than genuine categories. Returns the set of normalised
    sentinel tokens (lowercased, stripped). Empty set if the column does not look
    like 'numeric with a few placeholder codes'."""
    cfg = config.get("missingness", {}).get("sentinel_detection", {})
    if not cfg.get("enabled", True):
        return set()
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return set()
    as_str = non_missing.astype(str).str.strip()
    # Treat comma-decimals as numeric too (locale tolerance).
    numeric = pd.to_numeric(as_str.str.replace(",", ".", regex=False), errors="coerce")
    non_numeric_tokens = as_str[numeric.isna()]
    if non_numeric_tokens.empty:
        return set()
    # Candidate codes: the few distinct repeated non-numeric tokens.
    counts = non_numeric_tokens.str.lower().value_counts()
    counts = counts[(counts >= cfg.get("min_token_count", 2))
                    & (counts.index.str.len() <= cfg.get("max_token_length", 20))]
    if counts.empty or len(counts) > cfg.get("max_distinct_tokens", 5):
        return set()  # many distinct strings -> a real categorical, not codes
    # The codes must not cover essentially everything, and the REMAINDER must be
    # numeric (so the column is "numbers with a few repeated codes"). This catches
    # heavily-excluded measurements while never firing on genuine text columns.
    coded = non_missing.astype(str).str.strip().str.lower().isin(counts.index)
    coded_fraction = coded.mean()
    remainder_numeric_fraction = numeric[~coded.values].notna().mean() if (~coded.values).any() else 0.0
    if coded_fraction > cfg.get("max_token_fraction", 0.85):
        return set()
    if remainder_numeric_fraction < (1 - 1e-9) and numeric.notna().mean() < cfg.get("min_numeric_fraction", 0.20):
        return set()
    if remainder_numeric_fraction < 0.90:
        return set()  # the non-coded remainder isn't cleanly numeric -> treat as categorical
    # A real "measurement with placeholder codes" has a numeric remainder that is
    # much richer (more distinct values) than the handful of codes. If the numeric
    # part is itself low-cardinality (few distinct values), the column is more
    # likely a coded categorical where some levels are numbers and some are words,
    # so we do NOT treat the words as sentinels.
    distinct_numeric = int(numeric[~coded.values].dropna().nunique()) if (~coded.values).any() else 0
    if distinct_numeric < max(3, 2 * len(counts)):
        return set()
    # Exclude tokens already handled as technical/semantic missing markers.
    known = {normalize_name(x) for x in config["missingness"].get("technical_missing_markers", [])}
    known |= {normalize_name(x) for x in config["missingness"].get("semantic_missing_markers", [])}
    return {t for t in counts.index.tolist() if t not in known}


def numeric_series(series: pd.Series, config: Dict[str, Any]) -> pd.Series:
    """Coerce a column to float, treating both standard missing markers and any
    data-driven sentinel tokens as NaN, and tolerating comma decimals. Used by
    anomaly detection and role inference so placeholder codes never leak in as
    numbers or as false 'non-numeric value' accuracy issues."""
    s = series.astype(str).str.strip()
    sentinels = detect_sentinel_tokens(series, config)
    if sentinels:
        s = s.mask(s.str.lower().isin(sentinels))
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce").astype("float64")


def semantic_unusable_mask(series: pd.Series, config: Dict[str, Any]) -> pd.Series:
    markers = {normalize_name(x) for x in config["missingness"].get("semantic_missing_markers", [])}
    sentinels = detect_sentinel_tokens(series, config)
    flag_tokens = markers | sentinels
    if not flag_tokens:
        return pd.Series(False, index=series.index)
    return series.fillna("").astype(str).str.strip().str.lower().isin(flag_tokens)


def completeness_assessment(df: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Required/critical are honoured only when explicitly declared in metadata.
    # When absent, those components fall back to overall completeness (informative,
    # not a free 1.0) and no per-column "critical" triage spam is generated.
    required_declared = [c for c in metadata.get("required_fields", []) if c in df.columns]
    critical_declared = [c for c in metadata.get("critical_fields", []) if c in df.columns]
    required = required_declared or list(df.columns)
    critical = critical_declared or required
    rows = []
    for col in df.columns:
        technical = int(df[col].isna().sum())
        semantic = int(semantic_unusable_mask(df[col], config).sum())
        total = technical + semantic
        rows.append({
            "column": col,
            "technical_missing": technical,
            "semantic_unusable": semantic,
            "total_unusable": total,
            "technical_missing_rate": safe_div(technical, len(df)),
            "semantic_adjusted_unusable_rate": safe_div(total, len(df)),
            "semantic_adjustment_delta": safe_div(total - technical, len(df)),
        })

    miss = pd.DataFrame(rows).sort_values("semantic_adjusted_unusable_rate", ascending=False)
    technical_cell = 1 - safe_div(miss["technical_missing"].sum(), df.size)
    semantic_cell = 1 - safe_div(miss["total_unusable"].sum(), df.size)
    required_cell = 1 - safe_div(miss.loc[miss["column"].isin(required), "total_unusable"].sum(), len(df) * max(len(required), 1))
    critical_cell = 1 - safe_div(miss.loc[miss["column"].isin(critical), "total_unusable"].sum(), len(df) * max(len(critical), 1))
    row_mask = pd.DataFrame({col: df[col].isna() | semantic_unusable_mask(df[col], config) for col in df.columns})
    complete_rows = 1 - safe_div(row_mask.any(axis=1).sum(), len(df))

    weights = config["scoring"]["weights"]["C"]
    components = pd.DataFrame([
        {"component": "technical_cell", "score": clip01(technical_cell), "weight": weights["technical_cell"]},
        {"component": "semantic_cell", "score": clip01(semantic_cell), "weight": weights["semantic_cell"]},
        {"component": "required_fields", "score": clip01(required_cell), "weight": weights["required_fields"]},
        {"component": "critical_fields", "score": clip01(critical_cell), "weight": weights["critical_fields"]},
        {"component": "complete_rows", "score": clip01(complete_rows), "weight": weights["complete_rows"]},
    ])
    components["weighted_score"] = components["score"] * components["weight"]
    C = clip01(components["weighted_score"].sum())

    # Completeness issues -> HITL triage.
    # High-severity "critical_field_missingness" only when critical fields were
    # explicitly declared; otherwise emit informative per-variable missingness.
    issues = []
    critical_set = set(critical_declared)
    for _, r in miss.iterrows():
        if r["column"] in critical_set and r["semantic_adjusted_unusable_rate"] > 0:
            issues.append(issue("C", "completeness", "critical_field_missingness", r["column"],
                                int(r["total_unusable"]), "high", "review", "metadata critical_fields"))
        elif r["semantic_adjustment_delta"] > 0:
            issues.append(issue("C", "completeness", "semantic_unusable_values", r["column"],
                                int(r["semantic_unusable"]), "medium", "review", "config semantic_missing_markers"))
        elif r["semantic_adjusted_unusable_rate"] >= 0.20:
            issues.append(issue("C", "completeness", "high_missingness", r["column"],
                                int(r["total_unusable"]),
                                "high" if r["semantic_adjusted_unusable_rate"] >= 0.40 else "medium",
                                "review", "auto: >=20% missing"))
    return C, components, miss, pd.DataFrame(issues)


def missingness_baseline_comparison(miss: pd.DataFrame) -> pd.DataFrame:
    out = miss[[
        "column", "technical_missing", "semantic_unusable", "total_unusable",
        "technical_missing_rate", "semantic_adjusted_unusable_rate", "semantic_adjustment_delta",
    ]].copy()
    out["baseline"] = "generic_missingness_vs_semantic_adjustment"
    out["interpretation"] = np.where(
        out["semantic_adjustment_delta"] > 0,
        "Semantic missing markers increase the unusable-value burden for this variable.",
        "No semantic missingness adjustment for this variable.",
    )
    return out


# -----------------------------------------------------------------------------
# ACCURACY (A): structural + domain + inconsistency + semantic-aware anomalies
# -----------------------------------------------------------------------------


def structural_validity(df: pd.DataFrame, profile: pd.DataFrame, config: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
    rows = []
    low_conf = config["semantic_role_inference"].get("low_confidence_threshold", 0.65)
    for _, r in profile.iterrows():
        col, role = r["column"], r["role"]
        if role == "identifier":
            dups = int(df[col].notna().sum() - df[col].nunique(dropna=True))
            if dups > 0:
                rows.append(issue("A", "structural", "identifier_duplicate", col, dups, "high", "review"))
        if "numeric" in role or "measurement" in role:
            # Sentinel/placeholder codes are completeness (semantic-unusable), not
            # accuracy: count only genuinely-present values that still fail to
            # parse as numbers as true non-numeric anomalies.
            sentinels = detect_sentinel_tokens(df[col], config)
            sem = {normalize_name(x) for x in config["missingness"].get("semantic_missing_markers", [])}
            present = df[col].dropna().astype(str).str.strip()
            present = present[~present.str.lower().isin(sentinels | sem)]
            parsed = pd.to_numeric(present.str.replace(",", ".", regex=False), errors="coerce")
            non_numeric = int(parsed.isna().sum())
            if non_numeric > 0:
                rows.append(issue("A", "structural", "non_numeric_observed_value", col, non_numeric, "high", "review"))
        if float(r["role_confidence"]) < low_conf:
            rows.append(issue("A", "schema", "low_semantic_role_confidence", col, 1, "medium", "schema_confirmation"))
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        rows.append(issue("A", "structural", "duplicate_rows", None, duplicate_rows, "high", "review"))
    issues = pd.DataFrame(rows)
    burden = issues["count"].sum() if not issues.empty else 0
    return clip01(1 - safe_div(burden, df.size)), issues


def categorical_consistency(df: pd.DataFrame, profile: pd.DataFrame, config: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
    """Data-driven (no hardcoded categories): within a categorical column, detect
    distinct surface strings that collapse to the same canonical form after
    normalising case, whitespace and surrounding punctuation (e.g. 'Control',
    'Control(9)', 'Control (9)'). Such variants are an internal-consistency defect
    because they fragment one real category into several. Accuracy only; reuses
    the standard normalisation idea, introduces no new algorithm."""
    cat_roles = {"categorical_nominal", "ordinal_or_categorical", "free_text",
                 "free_text_or_high_cardinality", "binary"}
    low_card = config["semantic_role_inference"].get("low_cardinality_threshold", 30)
    issues = []
    checked_cols = 0
    inconsistent_cols = 0

    def canon(v: str) -> str:
        s = str(v).strip().lower()
        # Remove bracketed/parenthetical decorations together with their contents
        # (e.g. 'Control(9)', 'Control (9)' -> 'control'); these are annotations,
        # not distinct categories. Genuine categories like 'grade 1' vs 'grade 2'
        # keep their distinguishing token and are NOT merged.
        s = re.sub(r"[\(\[\{].*?[\)\]\}]", "", s)
        s = re.sub(r"[\s]+", "", s)               # collapse all whitespace
        s = re.sub(r"[\.\,\;\:\-_/\\]+", "", s)   # drop trailing punctuation/separators
        return s.strip()

    for _, r in profile.iterrows():
        col, role = r["column"], r["role"]
        if role not in cat_roles:
            continue
        values = df[col].dropna().astype(str).str.strip()
        if values.empty or values.nunique() > max(low_card, 50):
            continue
        checked_cols += 1
        groups: Dict[str, set] = {}
        for v in values.unique():
            groups.setdefault(canon(v), set()).add(v)
        collapsible = {k: vs for k, vs in groups.items() if len(vs) > 1}
        if collapsible:
            inconsistent_cols += 1
            n_variants = sum(len(vs) for vs in collapsible.values())
            example = "; ".join(" / ".join(sorted(vs)) for vs in list(collapsible.values())[:3])
            issues.append(issue("A", "inconsistency", "inconsistent_category_encoding", col,
                                n_variants, "medium", "review", f"variants collapse to same category: {example}"))
    score = clip01(1 - safe_div(inconsistent_cols, checked_cols)) if checked_cols else 1.0
    return score, pd.DataFrame(issues)


def domain_validity(df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Only metadata-supplied rules are used. No default clinical ranges exist."""
    valid_ranges = metadata.get("valid_ranges", {}) or {}
    allowed_values = metadata.get("allowed_values", {}) or {}
    issues, feature_rows = [], []
    for col in df.columns:
        checks = violations = 0
        if col in allowed_values:
            observed = df[col].dropna().astype(str).str.strip().str.lower()
            allowed = {normalize_name(v) for v in as_list(allowed_values[col])}
            bad = int((~observed.isin(allowed)).sum())
            checks += len(observed); violations += bad
            if bad > 0:
                issues.append(issue("A", "domain", "allowed_value_violation", col, bad, "high", "review", "metadata allowed_values"))
        if col in valid_ranges:
            low, high = valid_ranges[col]
            numeric = pd.to_numeric(df[col], errors="coerce")
            observed = numeric.notna()
            bad = int((observed & ((numeric < low) | (numeric > high))).sum())
            checks += int(observed.sum()); violations += bad
            if bad > 0:
                issues.append(issue("A", "domain", "range_violation", col, bad, "high", "review", f"metadata valid_ranges: [{low}, {high}]"))
        if checks > 0:
            feature_rows.append({"column": col, "domain_checks": checks, "domain_violations": violations,
                                 "domain_score": clip01(1 - safe_div(violations, checks))})
    feature = pd.DataFrame(feature_rows)
    score = feature["domain_score"].mean() if not feature.empty else 1.0
    return clip01(score), pd.DataFrame(issues), feature


def inconsistency_checks(df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
    """Cross-field consistency. Rules are metadata-supplied only:
    consistency_rules: [{"type":"not_greater_than","left":"admission","right":"discharge"},
                        {"type":"sum_equals","fields":[...],"total":"..."},
                        {"type":"non_negative","field":"..."}].
    If none provided, accuracy is unaffected (score 1.0, no issues)."""
    rules = metadata.get("consistency_rules", []) or []
    issues, total_checks, total_violations = [], 0, 0
    for rule in rules:
        rtype = rule.get("type")
        try:
            if rtype == "not_greater_than" and {"left", "right"} <= rule.keys():
                l = pd.to_numeric(df[rule["left"]], errors="coerce")
                r = pd.to_numeric(df[rule["right"]], errors="coerce")
                mask = l.notna() & r.notna()
                bad = int((mask & (l > r)).sum()); total_checks += int(mask.sum()); total_violations += bad
                if bad > 0:
                    issues.append(issue("A", "inconsistency", "left_greater_than_right",
                                        f"{rule['left']}>{rule['right']}", bad, "high", "review", "metadata consistency_rules"))
            elif rtype == "non_negative" and "field" in rule:
                x = pd.to_numeric(df[rule["field"]], errors="coerce")
                mask = x.notna()
                bad = int((mask & (x < 0)).sum()); total_checks += int(mask.sum()); total_violations += bad
                if bad > 0:
                    issues.append(issue("A", "inconsistency", "negative_value", rule["field"], bad, "medium", "review", "metadata consistency_rules"))
            elif rtype == "sum_equals" and {"fields", "total"} <= rule.keys():
                parts = sum(pd.to_numeric(df[f], errors="coerce") for f in rule["fields"])
                total = pd.to_numeric(df[rule["total"]], errors="coerce")
                mask = parts.notna() & total.notna()
                bad = int((mask & (np.abs(parts - total) > rule.get("tolerance", 1e-6))).sum())
                total_checks += int(mask.sum()); total_violations += bad
                if bad > 0:
                    issues.append(issue("A", "inconsistency", "sum_mismatch", rule["total"], bad, "medium", "review", "metadata consistency_rules"))
        except KeyError:
            continue
    score = clip01(1 - safe_div(total_violations, total_checks)) if total_checks > 0 else 1.0
    return score, pd.DataFrame(issues)


def numeric_columns_from_profile(df: pd.DataFrame, profile: pd.DataFrame, semantic: bool) -> List[str]:
    if not semantic:
        return [c for c in df.columns if pd.to_numeric(df[c].dropna(), errors="coerce").notna().mean() >= 0.98]
    return profile.loc[profile["role"].isin(ANOMALY_ELIGIBLE_ROLES), "column"].tolist()


def anomaly_flags(df: pd.DataFrame, columns: List[str], config: Dict[str, Any], proposed: bool
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Reused standard detectors. Key correction vs. a naive implementation:

    * IQR and modified z-score run on the RAW per-column values (NaN simply not
      flagged), so missing-driven anomalies are never silently 'repaired'.
    * Median fill is applied ONLY to the matrix fed to IsolationForest / LOF
      (which reject NaN). Its footprint is returned as imputation_report so the
      preprocessing is auditable and is not an undocumented repair step.
    """
    anomaly_cfg = config["anomaly_detection"]
    empty_report = pd.DataFrame(columns=["column", "imputed_cells", "imputed_rate"])
    if len(columns) < anomaly_cfg.get("min_numeric_columns", 2) or len(df) < anomaly_cfg.get("min_rows", 30):
        return pd.DataFrame(index=df.index), pd.DataFrame(), 1.0, empty_report

    # Coerce to float64 with sentinel/marker awareness: placeholder codes become
    # NaN (not spurious numbers), bool->0/1, comma-decimals handled. This keeps
    # detectors safe on any column type and avoids treating exclusion codes as
    # extreme values.
    raw = pd.DataFrame({c: numeric_series(df[c], config) for c in columns}, index=df.index)
    flags = pd.DataFrame(index=df.index)

    if proposed:
        iqr_multiplier = anomaly_cfg.get("iqr_multiplier", 3.0)
        mz_threshold = anomaly_cfg.get("modified_z_threshold", 4.5)
    else:
        base = anomaly_cfg.get("generic_baseline", {})
        iqr_multiplier = base.get("iqr_multiplier", 1.5)
        mz_threshold = base.get("modified_z_threshold", 3.5)

    # --- Statistical detectors on RAW values (no imputation) ---
    iqr_flag = pd.Series(False, index=df.index)
    mz_flag = pd.Series(False, index=df.index)
    for col in raw.columns:
        x = raw[col]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            iqr_flag = iqr_flag | (x < q1 - iqr_multiplier * iqr).fillna(False) | (x > q3 + iqr_multiplier * iqr).fillna(False)
        median = x.median()
        mad = np.median(np.abs(x.dropna() - median)) if x.notna().any() else 0
        if mad > 0:
            modified_z = 0.6745 * (x - median) / mad
            mz_flag = mz_flag | (np.abs(modified_z) > mz_threshold).fillna(False)
    flags["iqr"] = iqr_flag
    flags["modified_z"] = mz_flag

    # --- ML detectors require a complete matrix: median fill, reported ---
    imputed = raw.copy()
    report_rows = []
    for col in imputed.columns:
        n_missing = int(imputed[col].isna().sum())
        med = imputed[col].median()
        imputed[col] = imputed[col].fillna(0 if pd.isna(med) else med)
        report_rows.append({"column": col, "imputed_cells": n_missing, "imputed_rate": safe_div(n_missing, len(df))})
    imputation_report = pd.DataFrame(report_rows)

    ml_columns: List[str] = []
    x_scaled = StandardScaler().fit_transform(imputed)
    contamination = min(
        anomaly_cfg["isolation_forest"].get("contamination_max", 0.04),
        max(anomaly_cfg["isolation_forest"].get("contamination_min", 0.01),
            anomaly_cfg["isolation_forest"].get("contamination_expected_cases", 25) / len(df)),
    )
    if anomaly_cfg.get("isolation_forest", {}).get("enabled", True):
        flags["isolation_forest"] = IsolationForest(
            random_state=config["project"].get("random_seed", 42), contamination=contamination
        ).fit_predict(x_scaled) == -1
        ml_columns.append("isolation_forest")
    if anomaly_cfg.get("local_outlier_factor", {}).get("enabled", True):
        lof = anomaly_cfg["local_outlier_factor"]
        n_neighbors = min(lof.get("max_neighbors", 35),
                          max(lof.get("min_neighbors", 10), len(df) // lof.get("divisor_for_neighbors", 20)))
        flags["local_outlier_factor"] = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination
        ).fit_predict(x_scaled) == -1
        ml_columns.append("local_outlier_factor")

    method_cols = list(flags.columns)
    flags["method_count"] = flags[method_cols].sum(axis=1)
    if proposed:
        ens = anomaly_cfg.get("proposed_ensemble", {})
        min_methods = ens.get("min_methods", 2)
        if ens.get("require_ml_method", True) and ml_columns:
            flags["ensemble_flag"] = (flags["method_count"] >= min_methods) & flags[ml_columns].any(axis=1)
        else:
            flags["ensemble_flag"] = flags["method_count"] >= min_methods
    else:
        flags["ensemble_flag"] = flags[method_cols].any(axis=1)

    flags["row_index"] = df.index
    summary = flags[method_cols + ["ensemble_flag"]].sum().reset_index()
    summary.columns = ["method", "flagged_rows"]
    summary["flagged_rate"] = summary["flagged_rows"] / len(df)
    score = clip01(1 - safe_div(int(flags["ensemble_flag"].sum()), len(df)))
    return flags, summary, score, imputation_report


def anomaly_baseline_comparison(df: pd.DataFrame, semantic_profile: pd.DataFrame, uniform_profile: pd.DataFrame,
                                config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, pd.DataFrame]:
    generic_cols = numeric_columns_from_profile(df, uniform_profile, semantic=False)
    proposed_cols = numeric_columns_from_profile(df, semantic_profile, semantic=True)
    generic_flags, _, generic_score, _ = anomaly_flags(df, generic_cols, config, proposed=False)
    proposed_flags, _, proposed_score, imputation_report = anomaly_flags(df, proposed_cols, config, proposed=True)

    def flagged(f):
        return int(f["ensemble_flag"].sum()) if not f.empty and "ensemble_flag" in f else 0

    comparison = pd.DataFrame([
        {"method": "generic_anomaly_detection_all_numeric_columns",
         "numeric_columns_used": ",".join(generic_cols), "n_columns": len(generic_cols),
         "flagged_rows": flagged(generic_flags), "flagged_rate": safe_div(flagged(generic_flags), len(df)),
         "A_anomaly_score": generic_score, "hitl_prioritisation": "not_semantic"},
        {"method": "semantic_variable_aware_anomaly_detection_with_hitl_triage",
         "numeric_columns_used": ",".join(proposed_cols), "n_columns": len(proposed_cols),
         "flagged_rows": flagged(proposed_flags), "flagged_rate": safe_div(flagged(proposed_flags), len(df)),
         "A_anomaly_score": proposed_score, "hitl_prioritisation": "semantic_review_candidates"},
    ])
    return comparison, generic_flags, proposed_flags, generic_score, proposed_score, imputation_report


# -----------------------------------------------------------------------------
# REUSE READINESS (R): documentation + standardisation + privacy + machine-readability
# -----------------------------------------------------------------------------


def collect_documentation_text(metadata: Dict[str, Any]) -> str:
    parts = []
    for key in ["title", "description", "abstract", "dataset_type", "methodology", "provenance",
                "etl_description", "license", "usage_conditions", "standards", "vocabularies",
                "versioning", "reproducibility", "privacy", "documentation_text"]:
        value = metadata.get(key)
        if value:
            parts.append(f"{key}: {value}")
    for var, desc in (metadata.get("variable_descriptions", {}) or {}).items():
        if desc:
            parts.append(f"Variable {var}: {desc}")
    return "\n".join(parts)


_CE_STATE: Dict[str, Any] = {"tried": False, "model": None, "name": None}


def _get_cross_encoder(config: Dict[str, Any]):
    """ADVANCED LAYER 2 — lazily load a cross-encoder for documentation/prompt
    relevance re-ranking. Cached; returns None on any failure so callers fall
    back to bi-encoder cosine / TF-IDF. Deterministic given a fixed model."""
    ce_cfg = config["reuse_readiness"]["semantic_similarity"].get("cross_encoder", {})
    if not ce_cfg.get("enabled", False):
        return None
    if _CE_STATE["tried"]:
        return _CE_STATE["model"]
    _CE_STATE["tried"] = True
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        name = ce_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CE_STATE.update({"model": CrossEncoder(name), "name": name})
        print(f"AI-DQ3: cross-encoder reuse re-ranking enabled (model: {name}).")
        return _CE_STATE["model"]
    except Exception as exc:
        print(f"AI-DQ3: cross-encoder unavailable ({type(exc).__name__}); using bi-encoder/TF-IDF.")
        _CE_STATE["model"] = None
        return None


def _cross_encoder_doc_relevance(documentation: str, components: Dict[str, Any], config: Dict[str, Any]):
    """Return {component: (raw_logit, score_0_1)} via cross-encoder, or None."""
    model = _get_cross_encoder(config)
    if model is None or not documentation.strip() or not components:
        return None
    try:
        import numpy as _np
        ce_cfg = config["reuse_readiness"]["semantic_similarity"]["cross_encoder"]
        mid = float(ce_cfg.get("score_midpoint", 0.0))
        scale = float(ce_cfg.get("score_scale", 4.0))
        doc = documentation[:config["reuse_readiness"].get("llm_metadata_extraction", {}).get("max_documentation_chars", 12000)]
        pairs = [(doc, " ".join(v.get("prompts", []))) for v in components.values()]
        logits = model.predict(pairs)
        out = {}
        for component, lg in zip(components.keys(), _np.asarray(logits).ravel()):
            score = clip01(1.0 / (1.0 + _np.exp(-(float(lg) - mid) / max(scale, 1e-9))))
            out[component] = (float(lg), float(score))
        return out
    except Exception:
        return None


def _embedding_doc_similarity(documentation: str, components: Dict[str, Any], config: Dict[str, Any]):
    """Improvement 2 — semantic similarity between the documentation text and each
    reuse-readiness prompt using the embedding model. Returns a dict
    {component: normalised_score} or None if embeddings are unavailable."""
    model = _get_embedder(config)
    if model is None or not documentation.strip() or not components:
        return None
    try:
        import numpy as _np
        prompts = [" ".join(v.get("prompts", [])) for v in components.values()]
        doc_vec = model.encode([documentation], normalize_embeddings=True)[0]
        prompt_vecs = model.encode(prompts, normalize_embeddings=True)
        sims = _np.asarray(prompt_vecs) @ _np.asarray(doc_vec)
        nmin = config["reuse_readiness"]["semantic_similarity"].get("embedding_normalization_min", 0.15)
        nmax = config["reuse_readiness"]["semantic_similarity"].get("embedding_normalization_max", 0.55)
        out = {}
        for component, sim in zip(components.keys(), sims):
            out[component] = (float(sim), clip01((float(sim) - nmin) / max(nmax - nmin, 1e-9)))
        return out
    except Exception:
        return None


def text_similarity_scores(documentation: str, components: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    if not documentation.strip() or not components:
        for c in components.keys():
            rows.append({"component": c, "text_similarity": 0.0, "text_score": 0.0, "text_method": "not_applied_no_documentation"})
        return pd.DataFrame(rows)
    # Preference order: cross-encoder re-ranking > bi-encoder cosine > TF-IDF.
    ce = _cross_encoder_doc_relevance(documentation, components, config)
    if ce is not None:
        for component in components.keys():
            sim, score = ce[component]
            rows.append({"component": component, "text_similarity": sim, "text_score": score,
                         "text_method": "cross_encoder_reranking"})
        return pd.DataFrame(rows)
    emb = _embedding_doc_similarity(documentation, components, config)
    if emb is not None:
        for component in components.keys():
            sim, score = emb[component]
            rows.append({"component": component, "text_similarity": sim, "text_score": score,
                         "text_method": "embedding_cosine_similarity"})
        return pd.DataFrame(rows)
    prompts = [" ".join(v.get("prompts", [])) for v in components.values()]
    matrix = TfidfVectorizer(stop_words="english").fit_transform([documentation] + prompts)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    nmin = config["reuse_readiness"]["semantic_similarity"].get("normalization_min", 0.05)
    nmax = config["reuse_readiness"]["semantic_similarity"].get("normalization_max", 0.45)
    for component, sim in zip(components.keys(), sims):
        rows.append({"component": component, "text_similarity": float(sim),
                     "text_score": clip01((float(sim) - nmin) / max(nmax - nmin, 1e-9)),
                     "text_method": "tfidf_cosine_similarity"})
    return pd.DataFrame(rows)


def checklist_from_documentation_text(documentation: str, components: Dict[str, Any]) -> Dict[str, float]:
    """Structural keyword checklist: for each documentation/standardisation component,
    scan the collected documentation_text for any of its configured `checklist_keywords`.
    Returns {component: score} where score is 1.0 if at least one keyword matches,
    0.0 otherwise.

    This is the primary evidence source when no explicit metadata flag is set in the
    JSON metadata file.  It is intentionally generous (any single keyword hit = present)
    because the goal is to detect *existing* documentation — not to penalise imperfect
    wording.  The hybrid formula then blends this with text-similarity for the final score.
    """
    doc_lower = documentation.lower()
    scores: Dict[str, float] = {}
    for component, cfg in components.items():
        keywords = cfg.get("checklist_keywords", [])
        hit = any(kw.lower() in doc_lower for kw in keywords) if keywords else False
        scores[component] = 1.0 if hit else 0.0
    return scores


def hybrid_facet(components: Dict[str, Any], metadata: Dict[str, Any], documentation: str, config: Dict[str, Any],
                 facet_name: str = "documentation", confirmed: Optional[Dict[str, int]] = None
                 ) -> Tuple[float, pd.DataFrame, List[Dict[str, Any]]]:
    """Hybrid rule (metadata flag / structural checklist) + text-similarity scoring
    over a set of documentation/standardisation components.  Equal weight across
    components.

    Rule score resolution (in priority order):
      1. Explicit metadata flag in ``metadata["reuse_readiness"][component]`` or
         ``metadata[component]`` — honours manual/JSON overrides.
      2. Human-confirmed LLM proposal (``confirmed`` dict) — raises score to 1.0.
      3. Structural keyword checklist over ``documentation_text`` — detects README,
         codebook, licence, provenance etc. that are physically present in the
         metadata/ folder but not flagged in JSON metadata.

    The checklist contribution is controlled by ``reuse_readiness.checklist_rule_weight``
    in config (default 1.0 → checklist fully drives rule_score when no explicit flag).
    """
    confirmed = confirmed or {}
    reuse_meta = metadata.get("reuse_readiness", {}) or {}

    # Structural checklist scores derived from documentation text content.
    checklist_scores = checklist_from_documentation_text(documentation, components)
    crw = float(config.get("reuse_readiness", {}).get("checklist_rule_weight", 1.0))

    rule_rows = []
    missing_components = []
    for component in components.keys():
        # Priority 1: explicit metadata flag.
        explicit = reuse_meta.get(component, metadata.get(component))
        if explicit is not None:
            rule_score = score_value(explicit)
        else:
            # Priority 3: structural checklist (no explicit flag present).
            rule_score = clip01(checklist_scores.get(component, 0.0) * crw)
        # Priority 2: human-confirmed LLM proposal overrides everything.
        if rule_score < 1 and confirmed.get(component):
            rule_score = 1.0
        rule_rows.append({
            "component": component,
            "rule_score": rule_score,
            "rule_source": "metadata_flag" if explicit is not None else
                           ("llm_confirmed" if confirmed.get(component) else "checklist"),
        })
        if rule_score < 1:
            missing_components.append(component)

    rule_df = pd.DataFrame(rule_rows)
    text_df = text_similarity_scores(documentation, components, config)
    merged = rule_df.merge(text_df, on="component", how="left")

    # hybrid_score = checklist_weight * rule_score + similarity_weight * text_score
    # (per spec: 0.55 checklist + 0.45 embedding/text similarity). Config-driven.
    rw = config["reuse_readiness"].get("hybrid_checklist_weight", 0.55)
    tw = config["reuse_readiness"].get("hybrid_similarity_weight", 0.45)
    merged["hybrid_component_score"] = rw * merged["rule_score"] + tw * merged["text_score"].fillna(0.0)
    facet_score = clip01(merged["hybrid_component_score"].mean()) if not merged.empty else 0.0

    issues = []
    if missing_components:
        issues.append(issue("R", "metadata_checklist", f"{facet_name}_gaps", None,
                            len(missing_components),
                            "medium" if facet_score >= 0.34 else "high", "documentation",
                            "missing: " + ", ".join(missing_components)))
    return facet_score, merged, issues


def k_anonymity_assessment(df: pd.DataFrame, quasi_identifiers: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Standard k-anonymity measurement over declared quasi-identifiers.
    This is a measurement, not a new algorithm, and is metadata-driven."""
    qis = [c for c in quasi_identifiers if c in df.columns]
    k_min = config["reuse_readiness"]["privacy"].get("k_anonymity_min", 5)
    cap = config["reuse_readiness"]["privacy"].get("risk_fraction_cap", 0.50)
    if not qis or len(df) == 0:
        return {"assessable": False, "min_equivalence_class": None, "at_risk_fraction": None,
                "k_score": 0.0, "quasi_identifiers_used": qis}
    # Equivalence-class size per record across the quasi-identifier combination.
    group_sizes = df.groupby(qis, dropna=False)[qis[0]].transform("size")
    min_class = int(group_sizes.min())
    at_risk = float((group_sizes < k_min).mean())
    k_score = clip01(1 - min(at_risk, cap) / cap)  # 0 risk -> 1.0; risk>=cap -> 0.0
    return {"assessable": True, "min_equivalence_class": min_class, "at_risk_fraction": at_risk,
            "k_score": k_score, "quasi_identifiers_used": qis}


def privacy_facet(df: pd.DataFrame, metadata: Dict[str, Any], documentation: str, config: Dict[str, Any]
                  ) -> Tuple[float, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    cfg = config["reuse_readiness"]["privacy"]["components"]
    direct = [c for c in (metadata.get("direct_identifiers", []) or []) if c in df.columns]
    quasi = metadata.get("quasi_identifiers", []) or []
    auto = metadata.get("_auto_inferred", {}) or {}
    auto_direct = set(auto.get("direct_identifiers", []))
    auto_quasi = set(auto.get("quasi_identifiers", []))
    issues = []

    # 1) Direct identifier handling: present-in-data direct identifiers are high risk.
    if direct:
        direct_score = 0.0
        src = "auto-detected" if set(direct) & auto_direct else "metadata"
        issues.append(issue("R", "privacy", "direct_identifier_present", ",".join(direct),
                            len(direct), "high", "review", f"{src} direct identifiers"))
    else:
        direct_score = 1.0

    # 2) k-anonymity over quasi-identifiers (auto-detected when not declared).
    k = k_anonymity_assessment(df, quasi, config)
    if not k["assessable"]:
        k_score = 0.0
        issues.append(issue("R", "privacy", "no_quasi_identifiers_found", None, 1, "low",
                            "documentation", "no demographic/quasi-identifier columns detected"))
    else:
        k_score = k["k_score"]
        src = "auto-detected" if set(k["quasi_identifiers_used"]) & auto_quasi else "declared"
        k["source"] = src
        if k["at_risk_fraction"] and k["at_risk_fraction"] > 0:
            issues.append(issue("R", "privacy", "low_k_anonymity_records",
                                ",".join(k["quasi_identifiers_used"]),
                                int(round(k["at_risk_fraction"] * len(df))), "high", "review",
                                f"{src} QIs; records below k={config['reuse_readiness']['privacy'].get('k_anonymity_min', 5)}"))

    # 3) Privacy documentation present.
    priv_doc_score = score_value((metadata.get("reuse_readiness", {}) or {}).get(
        "privacy_documentation", 1.0 if ("privacy" in documentation.lower() or metadata.get("privacy")) else 0))
    if priv_doc_score < 1:
        issues.append(issue("R", "privacy", "privacy_documentation_gap", None, 1, "medium", "documentation", "privacy_documentation"))

    comp = pd.DataFrame([
        {"component": "direct_identifier_handling", "score": direct_score, "weight": cfg["direct_identifier_handling"]},
        {"component": "k_anonymity", "score": k_score, "weight": cfg["k_anonymity"]},
        {"component": "privacy_documentation", "score": priv_doc_score, "weight": cfg["privacy_documentation"]},
    ])
    comp["weighted_score"] = comp["score"] * comp["weight"]
    facet_score = clip01(comp["weighted_score"].sum())
    return facet_score, comp, issues, k


def machine_readability_facet(df: pd.DataFrame, semantic_profile: pd.DataFrame, encoding_note: str,
                              config: Dict[str, Any]) -> Tuple[float, pd.DataFrame, List[Dict[str, Any]]]:
    cfg = config["reuse_readiness"]["machine_readability"]["components"]
    low_conf = config["semantic_role_inference"].get("low_confidence_threshold", 0.65)
    issues = []

    confident = float((semantic_profile["role_confidence"] >= low_conf).mean()) if not semantic_profile.empty else 0.0
    unnamed = sum(1 for c in df.columns if str(c).lower().startswith("unnamed"))
    dup_cols = len(df.columns) - len(set(map(str, df.columns)))
    clean_headers = 1.0 if (unnamed == 0 and dup_cols == 0) else clip01(1 - safe_div(unnamed + dup_cols, len(df.columns)))
    ambiguous = float(semantic_profile["role"].isin(
        ["mixed_numeric_requires_review", "free_text_or_high_cardinality"]).mean()) if not semantic_profile.empty else 0.0
    parseable = clip01(1 - ambiguous)

    if encoding_note == "non_utf8_latin1_fallback":
        clean_headers *= 0.9
        issues.append(issue("R", "machine_readability", "non_utf8_encoding", None, 1, "low", "documentation", "encoding fallback"))
    if unnamed or dup_cols:
        issues.append(issue("R", "machine_readability", "unclean_headers", None, unnamed + dup_cols, "medium", "schema_confirmation", "headers"))

    comp = pd.DataFrame([
        {"component": "confident_typing", "score": clip01(confident), "weight": cfg["confident_typing"]},
        {"component": "clean_headers", "score": clip01(clean_headers), "weight": cfg["clean_headers"]},
        {"component": "parseable_values", "score": clip01(parseable), "weight": cfg["parseable_values"]},
    ])
    comp["weighted_score"] = comp["score"] * comp["weight"]
    return clip01(comp["weighted_score"].sum()), comp, issues


def llm_documentation_proposals(documentation: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """ADVANCED LAYER 3 — optional LLM extractor. Reads parsed documentation and
    proposes which reuse-readiness elements are present. Returns a dict with a
    per-element proposal (0/1) plus rationale, for HUMAN CONFIRMATION. Disabled by
    default; on any failure returns an empty, non-applied proposal. Never repairs
    data and never calls out unless explicitly enabled and configured."""
    cfg = config.get("reuse_readiness", {}).get("llm_metadata_extraction", {})
    elements = ["metadata_or_data_dictionary", "variable_descriptions", "units_or_value_labels",
                "licence_or_usage_conditions", "provenance_or_etl", "standards_or_vocabularies",
                "versioning_or_reproducibility", "privacy_documentation"]
    base = {"enabled": bool(cfg.get("enabled", False)), "provider": cfg.get("provider", "none"),
            "model": cfg.get("model", ""), "applied_to_checklist": bool(cfg.get("apply_to_checklist", False)),
            "status": "disabled", "proposals": {}, "note":
            "Proposals require human confirmation; not applied to scoring unless apply_to_checklist=true."}
    if not cfg.get("enabled", False) or not documentation.strip() or cfg.get("provider", "none") == "none":
        return base
    doc = documentation[: int(cfg.get("max_documentation_chars", 12000))]
    prompt = ("You are auditing dataset documentation for reuse readiness. For each element, answer "
              "present=1 or 0 with a one-line reason, as strict JSON {element: {present, reason}}. "
              f"Elements: {elements}. Documentation:\n{doc}")
    try:
        provider = cfg.get("provider")
        text = None
        if provider == "anthropic":
            import os, anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            msg = client.messages.create(model=cfg.get("model", "claude-3-5-haiku-latest"),
                                         max_tokens=800, messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        elif provider == "local":
            from transformers import pipeline as hf_pipeline  # type: ignore
            gen = hf_pipeline("text-generation", model=cfg.get("model"))
            text = gen(prompt, max_new_tokens=800)[0]["generated_text"]
        if text:
            import json as _json, re as _re
            m = _re.search(r"\{.*\}", text, _re.DOTALL)
            parsed = _json.loads(m.group(0)) if m else {}
            proposals = {e: {"present": int(bool(parsed.get(e, {}).get("present", 0))),
                             "reason": str(parsed.get(e, {}).get("reason", ""))[:200]} for e in elements}
            base.update({"status": "proposed", "proposals": proposals})
    except Exception as exc:
        base.update({"status": f"failed:{type(exc).__name__}"})
    return base


def reuse_readiness(df: pd.DataFrame, semantic_profile: pd.DataFrame, metadata: Dict[str, Any],
                    encoding_note: str, config: Dict[str, Any]
                    ) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    documentation = collect_documentation_text(metadata)
    rr = config["reuse_readiness"]
    weights = config["scoring"]["weights"]["R"]

    # ADVANCED LAYER 3: optional LLM proposals (human-confirmed). Applied to the
    # checklist only when apply_to_checklist is true.
    llm = llm_documentation_proposals(documentation, config)
    confirmed = {}
    if llm.get("applied_to_checklist") and llm.get("status") == "proposed":
        confirmed = {k: v["present"] for k, v in llm["proposals"].items() if v.get("present")}

    doc_score, doc_detail, doc_issues = hybrid_facet(rr["documentation_components"], metadata, documentation, config, "documentation", confirmed)
    std_score, std_detail, std_issues = hybrid_facet(rr["standardisation_components"], metadata, documentation, config, "standardisation", confirmed)
    priv_score, priv_detail, priv_issues, k_info = privacy_facet(df, metadata, documentation, config)
    mr_score, mr_detail, mr_issues = machine_readability_facet(df, semantic_profile, encoding_note, config)

    facets = pd.DataFrame([
        {"facet": "documentation", "score": doc_score, "weight": weights["documentation"]},
        {"facet": "standardisation", "score": std_score, "weight": weights["standardisation"]},
        {"facet": "privacy", "score": priv_score, "weight": weights["privacy"]},
        {"facet": "machine_readability", "score": mr_score, "weight": weights["machine_readability"]},
    ])
    facets["weighted_score"] = facets["score"] * facets["weight"]
    R = clip01(facets["weighted_score"].sum())

    detail = pd.concat([
        doc_detail.assign(facet="documentation"),
        std_detail.assign(facet="standardisation"),
        priv_detail.assign(facet="privacy"),
        mr_detail.assign(facet="machine_readability"),
    ], ignore_index=True)

    baseline = pd.DataFrame([
        {"method": "metadata_checklist_only",
         "R_score": clip01(weights["documentation"] * doc_detail["rule_score"].mean()
                           + weights["standardisation"] * std_detail["rule_score"].mean()
                           + weights["privacy"] * priv_score + weights["machine_readability"] * mr_score),
         "description": "Rule-based metadata completeness only (no text similarity)."},
        {"method": "hybrid_documentation_standardisation_privacy_machine_readability",
         "R_score": R,
         "description": "Four-facet hybrid: checklist + TF-IDF text similarity + k-anonymity + machine-readability."},
    ])
    issues = pd.DataFrame(doc_issues + std_issues + priv_issues + mr_issues)
    sim_method = detail["text_method"].dropna().iloc[0] if "text_method" in detail and detail["text_method"].notna().any() else "none"
    return R, facets, detail, baseline, {"issues": issues, "k_anonymity": k_info,
                                         "llm_proposals": llm, "similarity_method": sim_method}


# -----------------------------------------------------------------------------
# HITL TRIAGE LAYER (cross-cutting; NOT a scored dimension)
# -----------------------------------------------------------------------------


def _hitl_explanation(row: pd.Series, role_lookup: Dict[str, str]) -> str:
    """Improvement 3 — deterministic, template-based explanation for a flagged
    issue (no external LLM). Mentions issue type, affected column, semantic role,
    the triggered check, and why a human should review it."""
    issue_type = str(row.get("issue_type", "")).replace("_", " ")
    col = row.get("column")
    col = str(col) if (col is not None and str(col).lower() != "nan" and str(col) != "") else "(dataset-level)"
    role = role_lookup.get(str(row.get("column")), "unassigned/dataset-level")
    method = str(row.get("method", "")).replace("_", " ")
    rule = str(row.get("rule", "")) or "configured check"
    dim = {"C": "completeness", "A": "accuracy", "R": "reuse readiness"}.get(str(row.get("dimension", "")), "quality")
    n = int(row.get("count", 0))
    reasons = {
        "critical_field_missingness": "a clinically critical field is incomplete and may bias analysis",
        "semantic_unusable_values": "values use placeholder/exclusion codes that are unusable as numbers",
        "high_missingness": "the missing rate is high enough to affect downstream use",
        "non_numeric_observed_value": "a measurement column contains values that do not parse as numbers",
        "identifier_duplicate": "an identifier is expected to be unique but repeats",
        "duplicate_rows": "identical records may indicate accidental duplication",
        "range_violation": "values fall outside the metadata-defined plausible range",
        "allowed_value_violation": "values are outside the permitted category set",
        "inconsistent_category_encoding": "one real category is split across several spellings",
        "low_k_anonymity_records": "records are rare on quasi-identifiers, raising re-identification risk",
        "direct_identifier_present": "a direct personal identifier is present in the data",
        "ensemble_anomaly_candidate": "multiple detectors agree the record is anomalous",
        "low_semantic_role_confidence": "the variable's semantic role is uncertain",
    }
    reason = reasons.get(str(row.get("issue_type", "")), "the value needs expert judgement to confirm")
    return (f"{dim.capitalize()} issue '{issue_type}' on column '{col}' "
            f"(semantic role: {role}); triggered by {method} [{rule}], affecting {n} value(s). "
            f"Human review needed because {reason}; AI-DQ3 does not auto-correct data.")


def build_triage_register(issues: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any],
                          role_lookup: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Prioritise all candidate issues (across C, A, R) for expert review.
    Produces an ordered register; this is the operational HITL artefact. Each row
    carries a deterministic hitl_explanation (Improvement 3)."""
    role_lookup = role_lookup or {}
    if issues is None or issues.empty:
        return pd.DataFrame(columns=["priority_rank", "dimension", "issue_type", "column", "count",
                                     "severity", "priority_score", "suggested_decision_options", "hitl_explanation"])
    sev_rank = config["hitl"]["prioritisation"].get("severity_rank", {"high": 3, "medium": 2, "low": 1})
    boost = config["hitl"]["prioritisation"].get("critical_field_boost", 1)
    critical = {normalize_name(c) for c in metadata.get("critical_fields", [])}
    options = " | ".join(config["hitl"].get("allowed_decisions", []))

    reg = issues.copy()
    reg["priority_score"] = reg.apply(
        lambda r: sev_rank.get(str(r.get("severity", "")).lower(), 1)
        + (boost if normalize_name(r.get("column")) in critical else 0)
        + np.log1p(float(r.get("count", 0))), axis=1)
    reg = reg.sort_values("priority_score", ascending=False).reset_index(drop=True)
    reg["priority_rank"] = reg.index + 1
    reg["suggested_decision_options"] = options
    reg["hitl_explanation"] = reg.apply(lambda r: _hitl_explanation(r, role_lookup), axis=1)
    return reg[["priority_rank", "dimension", "method", "issue_type", "column", "count",
                "severity", "priority_score", "rule", "suggested_decision_options", "hitl_explanation"]]


def triage_summary(issues: pd.DataFrame) -> pd.DataFrame:
    if issues is None or issues.empty:
        return pd.DataFrame([{"dimension": "none", "n_issue_types": 0, "total_count": 0, "high_severity": 0}])
    g = issues.groupby("dimension").agg(
        n_issue_types=("issue_type", "count"),
        total_count=("count", "sum"),
        high_severity=("severity", lambda s: int((s.str.lower() == "high").sum())),
    ).reset_index()
    return g


def create_hitl_sample(df: pd.DataFrame, proposed_flags: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    sizes = config.get("hitl", {}).get("sample_size", {})
    n_anom, n_inc, n_ctrl = sizes.get("anomaly_rows", 30), sizes.get("incomplete_rows", 30), sizes.get("control_rows", 30)
    selected: List[Any] = []
    anomaly_idx = set()
    if proposed_flags is not None and not proposed_flags.empty and "ensemble_flag" in proposed_flags:
        anomaly_idx = set(proposed_flags.loc[proposed_flags["ensemble_flag"], "row_index"].tolist())
        selected += list(anomaly_idx)[:n_anom]
    incomplete = df.index[df.isna().any(axis=1)].tolist()
    selected += [i for i in incomplete if i not in selected][:n_inc]
    remaining = [i for i in df.index.tolist() if i not in selected]
    rng = np.random.default_rng(config["project"].get("random_seed", 42))
    if remaining:
        selected += rng.choice(remaining, size=min(n_ctrl, len(remaining)), replace=False).tolist()
    selected = list(dict.fromkeys(selected))

    rows = []
    for idx in selected:
        source = []
        if idx in anomaly_idx:
            source.append("semantic_ai_anomaly_candidate")
        if bool(df.loc[idx].isna().any()):
            source.append("incomplete_record")
        if not source:
            source.append("control_record")
        rows.append({
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "candidate_source": "+".join(source),
            "row_values_json": json.dumps(df.loc[idx].to_dict(), ensure_ascii=False, default=str),
            "manual_decision": "",
            "allowed_decisions": " | ".join(config.get("hitl", {}).get("allowed_decisions", [])),
            "reviewer": "", "rationale": "",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dimension aggregation, composite summary and sensitivity
# -----------------------------------------------------------------------------


def calculate_A(structural: float, domain: float, anomaly: float, inconsistency: float,
                schema_conf: float, config: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
    w = config["scoring"]["weights"]["A"]
    comp = pd.DataFrame([
        {"component": "structural", "score": clip01(structural), "weight": w["structural"]},
        {"component": "domain", "score": clip01(domain), "weight": w["domain"]},
        {"component": "anomaly", "score": clip01(anomaly), "weight": w["anomaly"]},
        {"component": "inconsistency", "score": clip01(inconsistency), "weight": w["inconsistency"]},
        {"component": "schema_confidence", "score": clip01(schema_conf), "weight": w["schema_confidence"]},
    ])
    comp["weighted_score"] = comp["score"] * comp["weight"]
    return clip01(comp["weighted_score"].sum()), comp


def composite_index(A: float, C: float, R: float, config: Dict[str, Any]) -> float:
    w = config["scoring"]["weights"]["composite"]
    return clip01(w["A"] * A + w["C"] * C + w["R"] * R)


def weight_sensitivity(A: float, C: float, R: float, config: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    threshold = config["scoring"]["thresholds"].get("composite", 0.85)
    for scenario, w in config.get("sensitivity_analysis", {}).get("scenarios", {}).items():
        score = clip01(w["A"] * A + w["C"] * C + w["R"] * R)
        rows.append({"scenario": scenario, "A_weight": w["A"], "C_weight": w["C"], "R_weight": w["R"],
                     "composite": score, "meets_threshold": bool(score >= threshold)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Controlled error injection (evaluation of the assessment layer)
# -----------------------------------------------------------------------------


def inject_controlled_errors(df: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any], rate: float
                             ) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(config["project"].get("random_seed", 42))
    corrupted = df.copy()
    truth = pd.Series(False, index=df.index)
    targets = {normalize_name(x) for x in metadata.get("target_fields", [])}
    numeric_cols = [c for c in df.columns if normalize_name(c) not in targets
                    and pd.to_numeric(df[c].dropna(), errors="coerce").notna().mean() >= 0.98]
    if not numeric_cols or len(df) == 0:
        return corrupted, truth
    # Cast the columns we will corrupt to float64 so bool/int columns can accept
    # NaN, sentinels and outlier values without dtype errors.
    for c in numeric_cols:
        corrupted[c] = pd.to_numeric(corrupted[c], errors="coerce").astype("float64")
    n = max(1, int(len(df) * rate))
    selected = rng.choice(df.index, size=min(n, len(df)), replace=False)
    error_types = config.get("baseline_experiment", {}).get("controlled_error_injection", {}).get(
        "error_types", ["set_missing", "numeric_sentinel", "high_outlier"])
    for i, idx in enumerate(selected):
        col = numeric_cols[i % len(numeric_cols)]
        etype = error_types[i % len(error_types)]
        truth.loc[idx] = True
        if etype == "set_missing":
            corrupted.loc[idx, col] = np.nan
        elif etype == "numeric_sentinel":
            corrupted.loc[idx, col] = -999.0
        elif etype == "high_outlier":
            q99 = pd.to_numeric(df[col], errors="coerce").astype("float64").quantile(0.99)
            corrupted.loc[idx, col] = q99 * 10 if pd.notna(q99) else 9999.0
        else:
            corrupted.loc[idx, col] = np.nan
    return corrupted, truth


def controlled_error_baseline(df: pd.DataFrame, metadata: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    if not config.get("baseline_experiment", {}).get("enabled", True):
        return pd.DataFrame()
    rates = config.get("baseline_experiment", {}).get("controlled_error_injection", {}).get("rates", [0.05])
    rows = []
    sp, up, _ = build_profiles(df, metadata, config)
    _, base_generic, base_proposed, _, _, _ = anomaly_baseline_comparison(df, sp, up, config)
    for rate in rates:
        corrupted, truth = inject_controlled_errors(df, metadata, config, rate)
        scp, ucp, _ = build_profiles(corrupted, metadata, config)
        _, generic_after, proposed_after, _, _, _ = anomaly_baseline_comparison(corrupted, scp, ucp, config)
        for method, before, after in [("generic_anomaly_delta", base_generic, generic_after),
                                       ("semantic_hitl_anomaly_delta", base_proposed, proposed_after)]:
            if before is None or before.empty or after is None or after.empty:
                pred = pd.Series(False, index=df.index); before_count = 0
            else:
                pred = after["ensemble_flag"] & (~before["ensemble_flag"]); before_count = int(before["ensemble_flag"].sum())
            p, r, f, _ = precision_recall_fscore_support(truth.astype(int), pred.astype(int), average="binary", zero_division=0)
            rows.append({"method": method, "injection_rate": rate, "precision": float(p), "recall": float(r),
                         "f1": float(f), "newly_flagged_rows": int(pred.sum()), "injected_error_rows": int(truth.sum()),
                         "pre_existing_flagged_rows": before_count})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# RQ2 consolidated comparison
# -----------------------------------------------------------------------------


def build_rq2_comparison(profile_comparison: pd.DataFrame, anomaly_comparison: pd.DataFrame,
                         missingness_comparison: pd.DataFrame, reuse_baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_roles_changed = int(profile_comparison["role_changed_by_semantic_layer"].sum())
    rows.append({"comparison_aspect": "variable_classification",
                 "uniform_view": f"{len(profile_comparison) - n_roles_changed} columns unchanged",
                 "semantic_view": f"{n_roles_changed} columns reclassified by meaning",
                 "effect": "Semantic roles change which checks apply to reclassified variables."})
    if anomaly_comparison is not None and len(anomaly_comparison) == 2:
        g, s = anomaly_comparison.iloc[0], anomaly_comparison.iloc[1]
        rows.append({"comparison_aspect": "accuracy_anomaly_detection",
                     "uniform_view": f"{int(g['n_columns'])} cols, {int(g['flagged_rows'])} rows flagged",
                     "semantic_view": f"{int(s['n_columns'])} cols, {int(s['flagged_rows'])} rows flagged",
                     "effect": "Semantic selection restricts detection to measurement-type variables and feeds HITL triage."})
    sem_adj = int((missingness_comparison["semantic_adjustment_delta"] > 0).sum()) if not missingness_comparison.empty else 0
    rows.append({"comparison_aspect": "completeness_missingness",
                 "uniform_view": "technical NaN only",
                 "semantic_view": f"{sem_adj} variables gain semantic-unusable burden",
                 "effect": "Semantic missing markers reprioritise completeness gaps."})
    if reuse_baseline is not None and len(reuse_baseline) == 2:
        rows.append({"comparison_aspect": "reuse_readiness",
                     "uniform_view": f"checklist-only R={reuse_baseline.iloc[0]['R_score']:.3f}",
                     "semantic_view": f"four-facet hybrid R={reuse_baseline.iloc[1]['R_score']:.3f}",
                     "effect": "Text similarity, privacy and machine-readability refine the reuse score."})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Per-dataset evaluation and outputs
# -----------------------------------------------------------------------------


def write_csv(path: Path, df: Optional[pd.DataFrame]) -> None:
    (df if df is not None else pd.DataFrame()).to_csv(path, index=False)


def evaluate_dataset(dataset_path: Path, metadata_dir: Path, output_dir: Path, config: Dict[str, Any],
                     n_datasets: int = 1) -> Dict[str, Any]:
    dataset_name = dataset_path.stem
    out = output_dir / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    raw, encoding_note = load_dataset(dataset_path)
    df = normalize_technical_missing(raw, config)
    metadata = find_related_metadata(dataset_path, metadata_dir, config, n_datasets=n_datasets)

    # --- Zero-config auto-inference: fill missing identifier/target/QI fields so
    #     ANY tabular dataset can be assessed without a metadata file. ---
    detected = auto_infer_metadata(df, metadata, config)
    metadata = apply_auto_inference(metadata, detected)
    out.mkdir(parents=True, exist_ok=True)
    (out / "auto_inferred_metadata.json").write_text(
        json.dumps(detected, indent=2, default=str), encoding="utf-8")

    # --- Semantic profiling (RQ1/RQ2 foundation) ---
    semantic_profile, uniform_profile, profile_comparison = build_profiles(df, metadata, config)
    check_selection = derive_check_selection(semantic_profile, metadata, config)

    # --- Completeness (C) ---
    C, C_components, missingness, C_issues = completeness_assessment(df, metadata, config)
    missingness_comparison = missingness_baseline_comparison(missingness)

    # --- Accuracy (A) ---
    A_structural, structural_issues = structural_validity(df, semantic_profile, config)
    A_domain, domain_issues, domain_features = domain_validity(df, metadata)
    A_rule_consistency, rule_inconsistency_issues = inconsistency_checks(df, metadata)
    A_cat_consistency, cat_inconsistency_issues = categorical_consistency(df, semantic_profile, config)
    # Inconsistency component combines metadata cross-field rules and data-driven
    # categorical-encoding consistency (both are "internal consistency").
    A_inconsistency = clip01(min(A_rule_consistency, A_cat_consistency))
    inconsistency_issues = pd.concat([rule_inconsistency_issues, cat_inconsistency_issues], ignore_index=True)
    anomaly_comparison, generic_flags, proposed_flags, _, A_anomaly, imputation_report = \
        anomaly_baseline_comparison(df, semantic_profile, uniform_profile, config)
    schema_confidence = float(semantic_profile["role_confidence"].mean()) if not semantic_profile.empty else 1.0
    A, A_components = calculate_A(A_structural, A_domain, A_anomaly, A_inconsistency, schema_confidence, config)

    # --- Reuse readiness (R) ---
    R, R_facets, R_detail, reuse_baseline, R_extra = reuse_readiness(df, semantic_profile, metadata, encoding_note, config)

    # --- Composite summary (convenience, not certification) ---
    composite = composite_index(A, C, R, config)
    th = config["scoring"]["thresholds"]
    sensitivity = weight_sensitivity(A, C, R, config)

    # --- HITL triage (cross-cutting, not scored) ---
    anomaly_issue_df = pd.DataFrame()
    if proposed_flags is not None and not proposed_flags.empty and "ensemble_flag" in proposed_flags \
            and proposed_flags["ensemble_flag"].sum() > 0:
        anomaly_issue_df = pd.DataFrame([issue("A", "semantic_ai_anomaly", "ensemble_anomaly_candidate", None,
                                               int(proposed_flags["ensemble_flag"].sum()), "medium", "review",
                                               "semantic anomaly candidates for HITL triage")])
    all_issues = pd.concat([C_issues, structural_issues, domain_issues, inconsistency_issues,
                            anomaly_issue_df, R_extra["issues"]], ignore_index=True)
    role_lookup = dict(zip(semantic_profile["column"].astype(str), semantic_profile["role"].astype(str)))
    triage_register = build_triage_register(all_issues, metadata, config, role_lookup=role_lookup)
    triage_stats = triage_summary(all_issues)
    hitl_sample = create_hitl_sample(df, proposed_flags, config)

    # --- Controlled error injection (evaluation) ---
    controlled_baseline = controlled_error_baseline(df, metadata, config)

    # --- RQ3 quality profile (the primary interpretable output) ---
    quality_profile = {
        "dataset": dataset_name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "C(D)": C, "A(D)": A, "R(D)": R, "composite_quality_index": composite,
        "A_structural": A_structural, "A_domain": A_domain, "A_anomaly": A_anomaly,
        "A_inconsistency": A_inconsistency, "schema_confidence": schema_confidence,
        "R_documentation": float(R_facets.loc[R_facets["facet"] == "documentation", "score"].iloc[0]),
        "R_standardisation": float(R_facets.loc[R_facets["facet"] == "standardisation", "score"].iloc[0]),
        "R_privacy": float(R_facets.loc[R_facets["facet"] == "privacy", "score"].iloc[0]),
        "R_machine_readability": float(R_facets.loc[R_facets["facet"] == "machine_readability", "score"].iloc[0]),
        "k_anonymity_min_class": R_extra["k_anonymity"].get("min_equivalence_class"),
        "k_anonymity_at_risk_fraction": R_extra["k_anonymity"].get("at_risk_fraction"),
        "reuse_similarity_method": R_extra.get("similarity_method", "none"),
        "C_meets_threshold": bool(C >= th.get("C", 0.95)),
        "A_meets_threshold": bool(A >= th.get("A", 0.95)),
        "R_meets_threshold": bool(R >= th.get("R", 0.80)),
        "composite_meets_threshold": bool(composite >= th.get("composite", 0.85)),
        "n_triage_candidates": int(len(triage_register)),
        "semantic_inference_method": ("embedding:" + str(_EMBED_STATE.get("name"))
                                      if _EMBED_STATE.get("model") is not None else "lexical_distributional_heuristic"),
    }
    quality_profile_df = pd.DataFrame([quality_profile])

    rq2 = build_rq2_comparison(profile_comparison, anomaly_comparison, missingness_comparison, reuse_baseline)

    # --- Write artifacts ---
    write_csv(out / "schema_profile.csv", semantic_profile)
    write_csv(out / "uniform_profile.csv", uniform_profile)
    write_csv(out / "rq1_check_selection_map.csv", check_selection)
    write_csv(out / "rq2_semantic_vs_uniform.csv", rq2)
    write_csv(out / "semantic_vs_uniform_profile.csv", profile_comparison)
    write_csv(out / "completeness_components.csv", C_components)
    write_csv(out / "missingness_comparison.csv", missingness_comparison)
    write_csv(out / "accuracy_components.csv", A_components)
    write_csv(out / "structural_issues.csv", structural_issues)
    write_csv(out / "domain_issues.csv", domain_issues)
    write_csv(out / "domain_feature_scores.csv", domain_features)
    write_csv(out / "inconsistency_issues.csv", inconsistency_issues)
    write_csv(out / "anomaly_baseline_comparison.csv", anomaly_comparison)
    write_csv(out / "anomaly_preprocessing_report.csv", imputation_report)
    write_csv(out / "generic_anomaly_flags.csv", generic_flags)
    write_csv(out / "semantic_hitl_anomaly_flags.csv", proposed_flags)
    write_csv(out / "reuse_facets.csv", R_facets)
    (out / "llm_documentation_proposals.json").write_text(
        json.dumps(R_extra.get("llm_proposals", {}), indent=2, default=str), encoding="utf-8")
    write_csv(out / "reuse_component_detail.csv", R_detail)
    write_csv(out / "reuse_baseline_comparison.csv", reuse_baseline)
    write_csv(out / "issues_all.csv", all_issues)
    write_csv(out / "hitl_triage_register.csv", triage_register)
    write_csv(out / "hitl_triage_summary.csv", triage_stats)
    write_csv(out / "hitl_validation_sample.csv", hitl_sample)
    write_csv(out / "controlled_error_baseline.csv", controlled_baseline)
    write_csv(out / "weight_sensitivity.csv", sensitivity)
    write_csv(out / "rq3_quality_profile.csv", quality_profile_df)
    (out / "rq3_quality_profile.json").write_text(json.dumps(quality_profile, indent=2, default=str), encoding="utf-8")

    summary = quality_profile_md(dataset_name, quality_profile, missingness, anomaly_comparison,
                                 reuse_baseline, controlled_baseline, sensitivity, triage_register, R_extra["k_anonymity"])
    (out / "quality_profile.md").write_text(summary, encoding="utf-8")
    return quality_profile


def quality_profile_md(dataset_name, q, missingness, anomaly_comparison, reuse_baseline,
                       controlled_baseline, sensitivity, triage_register, k_info) -> str:
    top_missing = missingness.sort_values("semantic_adjusted_unusable_rate", ascending=False).head(3)
    missing_text = "; ".join(f"{r['column']} ({r['semantic_adjusted_unusable_rate']:.1%})"
                             for _, r in top_missing.iterrows()) if not top_missing.empty else "no dominant pattern"
    anomaly_text = "; ".join(f"{r['method']} flagged {int(r['flagged_rows'])} rows ({r['flagged_rate']:.1%}) "
                             f"over {int(r['n_columns'])} variables"
                             for _, r in anomaly_comparison.iterrows()) + "." if anomaly_comparison is not None and not anomaly_comparison.empty else "n/a"
    reuse_text = "; ".join(f"{r['method']} R={r['R_score']:.3f}" for _, r in reuse_baseline.iterrows()) + "." \
        if reuse_baseline is not None and not reuse_baseline.empty else "n/a"
    if k_info.get("assessable"):
        privacy_text = (f"k-anonymity: minimum equivalence class = {k_info['min_equivalence_class']}, "
                        f"at-risk fraction = {k_info['at_risk_fraction']:.1%} over quasi-identifiers "
                        f"{k_info['quasi_identifiers_used']}.")
    else:
        privacy_text = "k-anonymity not assessable (no quasi-identifiers declared in metadata)."
    injection_text = "n/a"
    if controlled_baseline is not None and not controlled_baseline.empty:
        best = controlled_baseline.sort_values("f1", ascending=False).iloc[0]
        injection_text = (f"best delta detector = {best['method']} "
                          f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, F1={best['f1']:.3f}).")
    sens_text = f"composite ranges {sensitivity['composite'].min():.3f}–{sensitivity['composite'].max():.3f} across weighting scenarios." \
        if sensitivity is not None and not sensitivity.empty else "n/a"
    top_triage = "; ".join(f"#{int(r['priority_rank'])} {r['dimension']}/{r['issue_type']}"
                           f"{('/' + str(r['column'])) if pd.notna(r['column']) and r['column'] else ''} (n={int(r['count'])})"
                           for _, r in triage_register.head(5).iterrows()) if not triage_register.empty else "no candidates"

    return (
        f"# AI-DQ3 quality profile: {dataset_name}\n\n"
        f"Dataset: {int(q['rows'])} records x {int(q['columns'])} variables.\n\n"
        f"## Dataset-level quality profile (RQ3)\n"
        f"- Completeness  C(D) = {q['C(D)']:.3f}  (threshold met: {q['C_meets_threshold']})\n"
        f"- Accuracy      A(D) = {q['A(D)']:.3f}  (threshold met: {q['A_meets_threshold']})\n"
        f"  - structural={q['A_structural']:.3f}, domain={q['A_domain']:.3f}, anomaly={q['A_anomaly']:.3f}, "
        f"inconsistency={q['A_inconsistency']:.3f}, schema_confidence={q['schema_confidence']:.3f}\n"
        f"- Reuse readiness R(D) = {q['R(D)']:.3f}  (threshold met: {q['R_meets_threshold']})\n"
        f"  - documentation={q['R_documentation']:.3f}, standardisation={q['R_standardisation']:.3f}, "
        f"privacy={q['R_privacy']:.3f}, machine_readability={q['R_machine_readability']:.3f}\n"
        f"- Composite quality index = {q['composite_quality_index']:.3f} "
        f"(convenience summary, not a certification).\n\n"
        f"## Completeness\nMain semantic-adjusted missingness burden: {missing_text}.\n\n"
        f"## Accuracy\nAnomaly comparison (RQ2): {anomaly_text}\n\n"
        f"## Reuse readiness\nBaseline comparison: {reuse_text}\n{privacy_text}\n\n"
        f"## HITL triage (top candidates)\n{top_triage}.\nTotal triage candidates: {int(q['n_triage_candidates'])}.\n\n"
        f"## Evaluation\nControlled error injection: {injection_text}\nWeight robustness: {sens_text}\n\n"
        f"## Interpretation\nThese are pre-intervention data-quality diagnostics and HITL triage evidence. "
        f"They do not constitute autonomous clinical correction, automatic data repair, or legal compliance certification. "
        f"Privacy and re-identification risk are reported as components of reuse readiness, not as a separate legal assessment.\n"
    )


# -----------------------------------------------------------------------------
# Config resolution and entry point
# -----------------------------------------------------------------------------


def resolve_config_path(config_arg: Path) -> Path:
    candidates = []
    if config_arg.is_absolute():
        candidates.append(config_arg)
    else:
        candidates.append(Path.cwd() / config_arg)
        try:
            script_dir = Path(__file__).resolve().parent
            candidates += [script_dir / config_arg, script_dir.parent / config_arg]
        except NameError:
            pass
        candidates += [Path.cwd() / "ai_dq3_v2" / config_arg, Path("/content/ai_dq3_v2") / config_arg]
    for c in candidates:
        if c.exists():
            return c.resolve()
    searched = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        "config.yaml was not found. In Colab run:\n"
        "  %cd /content/ai_dq3_v2\n"
        "  !python pipeline.py --config config.yaml\n\n"
        f"Searched:\n{searched}"
    )


def run_pipeline(config_path: Path, data_dir: Optional[Path] = None,
                 metadata_dir: Optional[Path] = None, output_dir: Optional[Path] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    """Programmatic entry point (used by colab_bootstrap and notebooks).

    Runs the assessment on every tabular file in data_dir. A failure on one
    dataset does not abort the others — it is recorded and reported. Returns a
    summary dict with the output directory, the succeeded/failed datasets, and
    the per-dataset profiles."""
    config = read_config(config_path)
    root = config_path.parent.resolve()
    data_dir = Path(data_dir) if data_dir else root / config["input"].get("data_dir", "data")
    metadata_dir = Path(metadata_dir) if metadata_dir else root / config["input"].get("metadata_dir", "metadata")
    output_dir = Path(output_dir) if output_dir else root / config["project"].get("output_dir", "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = find_data_files(data_dir, config)
    max_datasets = config.get("project", {}).get("max_datasets")
    if max_datasets:
        data_files = data_files[: int(max_datasets)]

    summary: Dict[str, Any] = {"output_dir": str(output_dir), "data_dir": str(data_dir),
                               "succeeded": [], "failed": [], "profiles": []}
    if not data_files:
        summary["error"] = (f"No tabular data files found in '{data_dir}'. "
                            f"Place .csv/.xlsx/.xls files there and run again.")
        if verbose:
            print("AI-DQ3: " + summary["error"])
        return summary

    for p in data_files:
        try:
            profile = evaluate_dataset(p, metadata_dir, output_dir, config, n_datasets=len(data_files))
            summary["succeeded"].append(p.name)
            summary["profiles"].append(profile)
            if verbose:
                print(f"  [ok] {p.name}: C={profile.get('C(D)', float('nan')):.3f} "
                      f"A={profile.get('A(D)', float('nan')):.3f} R={profile.get('R(D)', float('nan')):.3f}")
        except Exception as exc:  # isolate per-dataset failures
            import traceback
            summary["failed"].append({"dataset": p.name, "error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()})
            if verbose:
                print(f"  [FAILED] {p.name}: {type(exc).__name__}: {exc}")

    if summary["profiles"]:
        pd.DataFrame(summary["profiles"]).to_csv(output_dir / "all_dataset_quality_profiles.csv", index=False)
    if verbose:
        print(f"AI-DQ3: {len(summary['succeeded'])} succeeded, {len(summary['failed'])} failed. "
              f"Results in {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI-DQ3 semantic variable-aware assessment pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring non-pipeline arguments injected by the environment: {unknown}")

    config_path = resolve_config_path(args.config)
    summary = run_pipeline(config_path, args.data_dir, args.metadata_dir, args.output_dir, verbose=True)
    if not summary["succeeded"] and summary.get("error"):
        raise FileNotFoundError(summary["error"])


if __name__ == "__main__":
    main()
