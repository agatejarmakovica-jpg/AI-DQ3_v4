# AI-DQ3: semantic variable-aware data quality assessment pipeline

AI-DQ3 is a single-version, configuration-driven pipeline for **semantic
variable-aware** quality assessment of **tabular healthcare datasets**. It uses
inferred variable meaning to guide three assessment dimensions —
**Completeness (C)**, **Accuracy (A)**, and **Reuse readiness (R)** — and adds a
cross-cutting **human-in-the-loop (HITL) triage** layer.

The pipeline does **not** introduce a new anomaly-detection, imputation, or
automatic data-repair algorithm. It reuses standard methods (IQR, modified
z-score, Isolation Forest, Local Outlier Factor, TF-IDF cosine similarity,
k-anonymity) and contributes the *semantic selection* of which checks apply,
*dimension-specific scoring*, and *HITL triage* of candidate issues.

## Scope and definitions

- **Completeness (C)** — presence and interpretation of missing values
  (technical missingness + semantic-unusable markers such as "not tested").
- **Accuracy (A)** — candidate errors, implausible values, anomalous patterns,
  duplicates, and inconsistencies (structural + domain + inconsistency +
  semantic-aware anomaly detection + schema confidence).
- **Reuse readiness (R)** — four facets: **documentation**, **standardisation**,
  **privacy-awareness** (incl. re-identification risk via k-anonymity over
  declared quasi-identifiers), and **machine-readability**. Privacy and
  re-identification risk are treated as components of reuse readiness, **not** as
  a separate legal-compliance assessment.
- **HITL triage** — a cross-cutting layer that prioritises candidate issues for
  expert review. It is **not** a scored dimension.

## Research-question → artifact mapping

For each dataset, results are written to `results/<dataset_name>/`:

| RQ  | Question | Primary artifact(s) |
|-----|----------|---------------------|
| RQ1 | How is semantic variable classification operationalised to **select** quality checks? | `rq1_check_selection_map.csv`, `schema_profile.csv` |
| RQ2 | How does semantic vs. uniform profiling affect **interpretation and prioritisation** of C/A/R issues? | `rq2_semantic_vs_uniform.csv`, `semantic_vs_uniform_profile.csv`, `missingness_comparison.csv`, `anomaly_baseline_comparison.csv`, `reuse_baseline_comparison.csv` |
| RQ3 | To what extent can AI-DQ3 generate an **interpretable dataset-level quality profile**? | `rq3_quality_profile.csv`, `rq3_quality_profile.json`, `quality_profile.md` |

Supporting artifacts: `completeness_components.csv`, `accuracy_components.csv`,
`reuse_facets.csv`, `reuse_component_detail.csv`, `issues_all.csv`,
`hitl_triage_register.csv`, `hitl_triage_summary.csv`,
`hitl_validation_sample.csv`, `anomaly_preprocessing_report.csv` (imputation
footprint transparency), `controlled_error_baseline.csv` (precision/recall/F1
validation), `weight_sensitivity.csv`.

## Optional: embedding-based semantic inference

By default, semantic roles are inferred from variable names, value distributions
and (optional) metadata — lexical + distributional inference, with a confidence
reported per variable. This needs no extra dependency and is fully deterministic.

For stronger, meaning-based matching (synonyms, unseen names, partial other
languages), an **optional** embedding layer can replace the name-pattern step:
sentence-transformers encodes each variable's name (+ description + sample
values) and matches it against natural-language role prototypes by cosine
similarity. To enable it:

```bash
pip install -r requirements-semantic.txt
# then set in config.yaml:
#   semantic_role_inference:
#     embedding:
#       enabled: true
```

It is **off by default** to keep the pipeline dependency-free and reproducible.
If enabled but the library or model is unavailable (e.g. offline), the pipeline
automatically falls back to the heuristics — it never crashes. The method used
(`embedding:<model>` or `lexical_distributional_heuristic`) and the model name
are recorded in `rq3_quality_profile.csv` for reproducibility. Heavier options
(scispaCy/UMLS clinical-concept mapping, zero-shot LLM classification) are
deliberately not bundled: they add licensing/size/non-determinism that would
undermine the reproducibility the study relies on.

## Zero-config: works on any tabular dataset

AI-DQ3 assesses **any tabular healthcare dataset with no metadata at all**. When
metadata is absent or partial, it auto-detects identifiers, direct identifiers,
quasi-identifiers (for the privacy / k-anonymity facet) and a likely target from
the data, and writes them to `results/<dataset>/auto_inferred_metadata.json` for
review.

### Uploading a bundle (dataset + codebook + metadata)

You can upload a **ZIP** containing the dataset plus any supporting files. The
Colab runner unpacks it and routes files automatically:

- tabular files (`.csv/.xlsx/.xls/.tsv`) → `data/`
- structured metadata (`<stem>_metadata.json`) → `metadata/` (merged)
- documentation — **codebook / data dictionary / readme**, including `.pdf`,
  `.docx`, `.txt`, `.md` → `metadata/` and parsed to text.

Parsed documentation is used by the reuse-readiness facets: an uploaded codebook
or data dictionary **legitimately raises the documentation score** (via TF-IDF
similarity against the documentation prototypes), so you do not have to hand-fill
the `reuse_readiness` checklist to get credit for documentation you already have.
Documentation files are matched to a dataset by shared filename tokens, by
documentation name hints (readme/codebook/dictionary), or — when a single
dataset is present — attached to it directly. PDF/DOCX parsing degrades
gracefully if the readers are not installed.

Metadata is an **optional enhancement**, never a precondition. `pipeline.py`
contains no dataset-specific schema, clinical range, allowed-value list, target,
required/critical field, or quasi-identifier — these come from optional metadata
JSON, and generic thresholds/weights live in `config.yaml`.

To improve accuracy and reuse-readiness scoring, add an optional metadata file
named after the dataset (`data/my_cohort.csv` → `metadata/my_cohort_metadata.json`).
The template `metadata/metadata_template.json` is fully generic (no fixed column
names — every field is optional and refers to your own columns); see
`metadata/METADATA_GUIDE.md` for the field reference, the semantic-role
vocabulary, and worked examples for several different healthcare dataset shapes
in `metadata/examples/`.

## Transparency of preprocessing

Median fill is applied **only** to the matrix passed to Isolation Forest / LOF
(which reject NaN). IQR and modified z-score operate on raw, non-imputed values,
so missing-driven anomalies are never silently repaired. The imputation
footprint is reported in `anomaly_preprocessing_report.csv`. The dataset
returned by the pipeline is never modified.

## Local use

```bash
pip install -r requirements.txt
# place CSV/XLSX in data/ and optional metadata JSON in metadata/
python pipeline.py --config config.yaml
python figures.py --results results --out figures   # manuscript-ready figures (no titles)
python report.py  --results results --figures figures --out results   # consolidated HTML report
```

## Consolidated results report

`report.py` writes `results/<dataset>/report.html` — a single self-contained
page (figures embedded) that:

- states the **study aim** and a verdict on whether it is **demonstrated** for
  the dataset, via explicit capability criteria (RQ1, RQ2, RQ3, HITL, evaluation,
  no-repair);
- gives **auto-derived answers to RQ1, RQ2 and RQ3** computed from the result
  CSVs (not hand-written);
- shows the C/A/R profile, the HITL triage table, and all figures with suggested
  captions.

It also writes `aim_assessment.json` (machine-readable verdict). The report
distinguishes **method capability** (what the aim/RQs ask) from the **dataset's
own quality scores** — a dataset can score low while the method still fully
demonstrates the aim.

## Publication figures

`figures.py` reads the per-dataset result CSVs and writes vector PDF + 300 dpi
PNG figures to `figures/<dataset>/` using a colour-blind-safe Okabe-Ito palette
with a consistent dimension encoding (Completeness = blue, Accuracy =
vermillion, Reuse readiness = green; baseline = grey, proposed = blue).

Figures are **title-free by default** so they can be pasted directly into a
manuscript (captions belong in the paper text); pass `--titles` to embed titles
for slides or quick review. The overview keeps its (a)–(d) panel letters.

- `fig_overview` — multi-panel manuscript Figure 1 (profile, RQ2 anomaly
  comparison, missingness, controlled-error validation).
- `fig_quality_profile` — C/A/R + composite with acceptance thresholds (RQ3).
- `fig_component_breakdown` — sub-components of each dimension.
- `fig_rq2_anomaly_comparison` — uniform vs semantic detection (RQ2).
- `fig_completeness_missingness` — per-variable technical vs semantic-adjusted.
- `fig_hitl_triage` — top triage candidates coloured by dimension.
- `fig_controlled_error_validation` — precision/recall/F1 of detectors.
- `fig_weight_sensitivity` — composite index across weighting scenarios.

PDF output is vector (lossless) and is the format most Q1 journals request for
line-art figures.

## Colab use

See `RUN_COLAB.txt`.

## Limitations (for the manuscript)

- Semantic role inference is lexical + distributional, not deep semantic
  understanding; confidence is reported per variable and low-confidence roles
  are routed to HITL.
- TF-IDF cosine similarity measures lexical overlap with reference prompts, not
  meaning; multilingual or synonym-rich documentation may score lower.
- k-anonymity is a measurement over declared quasi-identifiers; if none are
  declared, the privacy facet flags this as a documentation gap rather than
  assuming safety.
- The composite quality index is a configurable convenience summary, not a
  certification; the primary output is the interpretable (C, A, R) vector with
  component breakdowns.

## Data-driven robustness corrections (no hardcoding)

Three generic, data-driven corrections improve assessment fidelity on diverse
healthcare tables without any dataset-specific rules:

1. **Sentinel / placeholder detection (completeness).** In a column whose
   non-coded values are cleanly numeric and rich in distinct values, a small set
   of repeated non-numeric tokens (e.g. exclusion/placeholder codes) is detected
   statistically and treated as *semantic-unusable* (completeness), not as
   non-numeric accuracy errors. No token wordlist is used; thresholds are generic
   (`missingness.sentinel_detection` in config.yaml). The same sentinel-aware
   parsing feeds role inference and anomaly detection, so placeholder codes never
   masquerade as numbers or as false anomalies.

2. **Key vs. measurement disambiguation (role inference).** A column is only
   labelled `identifier` when it is near-unique *and* key-like (integer codes or
   string keys). High-resolution continuous measurements that happen to be
   near-unique (e.g. a lung-function percentage) are no longer mislabelled as
   identifiers, which also removes spurious duplicate-identifier flags.

3. **Categorical-encoding consistency (accuracy).** Distinct surface strings that
   collapse to the same canonical form after removing bracketed annotations and
   whitespace (e.g. `Control`, `Control(9)`, `Control (9)`) are flagged as an
   internal-consistency defect. Genuinely distinct categories (e.g. `grade 1` vs
   `grade 2`) are preserved. No category list is hardcoded.

These were validated on a real University of Latvia Biobank lung-cancer dataset:
completeness correctly dropped once hidden exclusion codes were counted, the
categorical inconsistency was detected, and continuous measurements were no
longer mistaken for identifiers — while a regression battery (all-boolean,
all-text, coded-categorical, clinical) showed no false positives.

## AI-assisted components (summary)

AI-DQ3 is AI-assisted through three optional, additive components that preserve
the existing structure, formulas and outputs:

1. **Semantic role inference.** Each column is matched against predefined role
   prototypes with embeddings (`sentence-transformers/all-MiniLM-L6-v2`); the
   schema profile reports `semantic_role_rule_based`, `semantic_role_embedding`,
   `semantic_role_final`, `semantic_role_confidence`, `semantic_role_source`,
   and `requires_semantic_review`. Falls back to rule-based logic if the library
   is not installed.
2. **Reuse-readiness documentation interpretation.** Embedding (or TF-IDF
   fallback) similarity between documentation text and reuse prompts, combined as
   `hybrid = 0.55*checklist + 0.45*similarity` (config-driven). Reuse readiness is
   not purely AI-based.
3. **Explainable HITL triage.** Every flagged issue carries a deterministic
   `hitl_explanation` (issue type, column, semantic role, triggered check, reason
   for review). No external LLM is used by default; data is never auto-repaired.

**Files changed:** `pipeline.py` (functions added: `role_provenance`,
`_embedding_doc_similarity`, `_hitl_explanation`; updated: `build_profiles`,
`text_similarity_scores`, `hybrid_facet`, `build_triage_register`), `config.yaml`,
`README.md`.

**Config options added:** `reuse_readiness.semantic_similarity.embedding_normalization_min/max`,
`reuse_readiness.hybrid_checklist_weight` (0.55), `reuse_readiness.hybrid_similarity_weight` (0.45);
existing `semantic_role_inference.embedding.*` controls the role model.

**How to run:** rule-based by default (`%run colab_bootstrap.py`). To enable the
embedding mode: `pip install -r requirements-semantic.txt` and set
`semantic_role_inference.embedding.enabled: true` in `config.yaml`.

**Unchanged:** composite remains `0.40*A + 0.30*C + 0.30*R`; HITL is not a score
dimension; no dataset-specific columns are hard-coded; rule-based mode still works;
outputs remain compatible (new columns are additive).

**Limitations:** embeddings need a one-time model download (offline → automatic
rule-based fallback); the embedding model is general-domain (not clinical); HITL
explanations are template-based, not generative; similarity scoring measures
lexical/semantic overlap, not factual correctness of documentation.

## Advanced AI layers (optional, graceful fallback)

Three optional layers strengthen the semantic/interpretation stage without
touching the C/A/R formulas or the composite (`0.40*A + 0.30*C + 0.30*R`). All
are off by default and degrade to the existing logic if a model/library is
unavailable.

**Layer 1 — domain embeddings for semantic role inference.** The role-inference
model is swappable via `semantic_role_inference.embedding.model`. For biomedical
text, set e.g. `pritamdeka/S-PubMedBert-MS-MARCO` or `FremyCompany/BioLORD-2023`;
the general MiniLM default works for any domain. Requires `requirements-semantic.txt`.

**Layer 2 — cross-encoder re-ranking of reuse documentation.** When
`reuse_readiness.semantic_similarity.cross_encoder.enabled: true`, documentation
is scored against each reuse prompt with a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), which judges relevance more accurately
than bi-encoder cosine. Preference order: cross-encoder → bi-encoder cosine →
TF-IDF. The method used is recorded as `reuse_similarity_method` in the profile.

**Layer 3 — optional LLM metadata extractor (human-confirmed).** When
`reuse_readiness.llm_metadata_extraction.enabled: true`, an LLM reads the parsed
README/codebook and proposes which reuse elements are present, written to
`results/<ds>/llm_documentation_proposals.json`. Proposals raise the checklist
score ONLY if `apply_to_checklist: true` (i.e. after human confirmation). This
fixes the low-R problem for well-documented datasets at its root. Requires
`requirements-llm.txt`; provider is `anthropic` (needs `ANTHROPIC_API_KEY`) or a
local Hugging Face model. Never repairs data; no external call unless enabled.

**Config added:** `semantic_role_inference.embedding.model` (domain models),
`reuse_readiness.semantic_similarity.cross_encoder.*`,
`reuse_readiness.llm_metadata_extraction.*`. **New outputs:**
`llm_documentation_proposals.json`, `reuse_similarity_method` in the profile.
**Limitations:** cross-encoder/embeddings need a one-time model download (offline
→ TF-IDF fallback); LLM extraction is non-deterministic and must be
human-confirmed before it affects scoring; composite, dimensions and HITL status
are unchanged.
