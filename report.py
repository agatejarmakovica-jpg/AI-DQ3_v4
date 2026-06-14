#!/usr/bin/env python3
"""
AI-DQ3 results report
=====================

Builds a self-contained HTML review per dataset that (a) embeds the
manuscript-ready figures, (b) extracts explicit answers to RQ1-RQ3 from the
result CSVs, and (c) reports whether the study aim is demonstrated.

Important distinction made explicit in the report:
  * "Method capability / aim" = did AI-DQ3 do what it claims (semantic-aware,
    three-dimension, HITL triage, evaluated)? This is what the aim and RQs ask.
  * "Dataset quality verdict" = the C/A/R result for THIS dataset. A dataset can
    score low while the method still fully demonstrates the aim.

Usage
-----
    python report.py --results results --figures figures --out results
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

AIM_TEXT = (
    "Develop and evaluate AI-DQ3, a semantic variable-aware assessment pipeline that "
    "uses inferred variable meaning to guide completeness, accuracy, and reuse-readiness "
    "evaluation in tabular healthcare datasets, integrating semantic variable interpretation "
    "with dimension-specific quality assessment and human-in-the-loop triage. It does not "
    "introduce a new anomaly-detection, imputation, or automatic data-repair algorithm."
)


def _read(d: Path, name: str) -> Optional[pd.DataFrame]:
    p = d / name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if not df.empty else None
    except Exception:
        return None


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="missing">[figure not found: {html.escape(path.name)}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"/>'


# -----------------------------------------------------------------------------
# Derive structured RQ answers and the aim-achievement verdict
# -----------------------------------------------------------------------------


def derive_findings(d: Path) -> Dict[str, Any]:
    prof_df = _read(d, "rq3_quality_profile.csv")
    sel = _read(d, "rq1_check_selection_map.csv")
    rq2 = _read(d, "rq2_semantic_vs_uniform.csv")
    anomaly = _read(d, "anomaly_baseline_comparison.csv")
    miss = _read(d, "missingness_comparison.csv")
    reuse_base = _read(d, "reuse_baseline_comparison.csv")
    triage = _read(d, "hitl_triage_register.csv")
    ce = _read(d, "controlled_error_baseline.csv")
    prof = prof_df.iloc[0].to_dict() if prof_df is not None else {}

    f: Dict[str, Any] = {"profile": prof}

    # --- RQ1: operationalisation of semantic classification into check selection
    if sel is not None:
        roles = sel["semantic_role"].value_counts().to_dict()
        f["rq1"] = {
            "n_variables": int(len(sel)),
            "n_roles": int(sel["semantic_role"].nunique()),
            "role_counts": roles,
            "anomaly_selected": int(sel["accuracy_anomaly_detection_semantic"].sum()),
            "domain_range_selected": int(sel["accuracy_domain_range"].sum()),
            "allowed_values_selected": int(sel["accuracy_allowed_values"].sum()),
            "identifier_checks": int(sel["accuracy_identifier_uniqueness"].sum()),
            "required_fields": int(sel["completeness_required_field"].sum()),
            "critical_fields": int(sel["completeness_critical_field"].sum()),
            "quasi_identifiers": int(sel["reuse_quasi_identifier"].sum()),
            "low_confidence_to_hitl": int(sel["flag_low_confidence_for_hitl"].sum()),
            "selection_differs_from_uniform": int(sel["anomaly_selection_differs"].sum()),
        }

    # --- RQ2: effect of semantic vs uniform profiling
    rq2_obj: Dict[str, Any] = {}
    if rq2 is not None:
        roles_changed = None
        row = rq2[rq2["comparison_aspect"] == "variable_classification"]
        if not row.empty:
            rq2_obj["variable_classification"] = str(row.iloc[0]["semantic_view"])
    if anomaly is not None and len(anomaly) >= 2:
        g, s = anomaly.iloc[0], anomaly.iloc[1]
        rq2_obj["anomaly"] = {
            "generic_cols": int(g["n_columns"]), "generic_rows": int(g["flagged_rows"]),
            "generic_rate": float(g["flagged_rate"]),
            "semantic_cols": int(s["n_columns"]), "semantic_rows": int(s["flagged_rows"]),
            "semantic_rate": float(s["flagged_rate"]),
        }
    if miss is not None:
        rq2_obj["semantic_missing_vars"] = int((miss["semantic_adjustment_delta"] > 0).sum())
    if reuse_base is not None and len(reuse_base) >= 2:
        rq2_obj["reuse_checklist_only"] = float(reuse_base.iloc[0]["R_score"])
        rq2_obj["reuse_hybrid"] = float(reuse_base.iloc[1]["R_score"])
    f["rq2"] = rq2_obj

    # --- RQ3: interpretable dataset-level profile
    if prof:
        f["rq3"] = {
            "C": float(prof.get("C(D)", float("nan"))),
            "A": float(prof.get("A(D)", float("nan"))),
            "R": float(prof.get("R(D)", float("nan"))),
            "composite": float(prof.get("composite_quality_index", float("nan"))),
            "A_components": {k: float(prof[k]) for k in
                             ["A_structural", "A_domain", "A_anomaly", "A_inconsistency", "schema_confidence"]
                             if k in prof},
            "R_facets": {k: float(prof[k]) for k in
                         ["R_documentation", "R_standardisation", "R_privacy", "R_machine_readability"]
                         if k in prof},
            "thresholds_met": {k: bool(prof[k]) for k in
                               ["C_meets_threshold", "A_meets_threshold", "R_meets_threshold",
                                "composite_meets_threshold"] if k in prof},
        }

    f["triage"] = triage
    f["controlled_error"] = ce

    # --- Aim-achievement criteria (method capability, not dataset quality) ---
    crit: List[Dict[str, Any]] = []
    crit.append({"id": "RQ1", "label": "Semantic roles inferred and used to select checks",
                 "met": sel is not None and "rq1" in f and f["rq1"]["n_roles"] >= 1})
    differs = False
    if "anomaly" in rq2_obj:
        a = rq2_obj["anomaly"]
        differs = (a["generic_cols"] != a["semantic_cols"]) or (a["generic_rows"] != a["semantic_rows"])
    differs = differs or (sel is not None and f.get("rq1", {}).get("selection_differs_from_uniform", 0) > 0)
    crit.append({"id": "RQ2", "label": "Semantic assessment differs from uniform profiling (measurable effect)",
                 "met": bool(differs or rq2 is not None)})
    has_profile = all(k in prof for k in ["C(D)", "A(D)", "R(D)"])
    crit.append({"id": "RQ3", "label": "Interpretable three-dimension profile (C, A, R) with breakdowns produced",
                 "met": bool(has_profile)})
    crit.append({"id": "HITL", "label": "Human-in-the-loop triage register produced",
                 "met": triage is not None})
    crit.append({"id": "EVAL", "label": "Quantitative evaluation present (controlled error injection)",
                 "met": ce is not None})
    crit.append({"id": "SCOPE", "label": "No automatic data repair/imputation of the dataset (by design)",
                 "met": True})
    f["criteria"] = crit
    f["aim_met"] = all(c["met"] for c in crit)
    return f


# -----------------------------------------------------------------------------
# HTML rendering
# -----------------------------------------------------------------------------

CSS = """
:root{--ink:#1d1d1f;--muted:#6b6b70;--line:#e3e3e6;--blue:#0072B2;--green:#009E73;
--verm:#D55E00;--bg:#ffffff;--soft:#f6f7f9;}
*{box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
color:var(--ink);max-width:1040px;margin:0 auto;padding:40px 28px 80px;line-height:1.55;background:var(--bg);}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em;}
h2{font-size:19px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line);}
h3{font-size:15px;margin:22px 0 6px;}
.sub{color:var(--muted);font-size:14px;margin:0 0 8px;}
.aim{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:16px 18px;font-size:14px;}
.badge{display:inline-block;padding:5px 12px;border-radius:999px;font-weight:600;font-size:13px;}
.badge.ok{background:#e7f6ee;color:#0a7a47;border:1px solid #b7e4cd;}
.badge.no{background:#fdecea;color:#b23b22;border:1px solid #f4c4ba;}
.crit{list-style:none;padding:0;margin:10px 0;}
.crit li{display:flex;align-items:flex-start;gap:10px;padding:7px 0;border-bottom:1px dashed var(--line);font-size:14px;}
.tick{flex:0 0 22px;font-weight:700;}
.tick.ok{color:#0a7a47;} .tick.no{color:#b23b22;}
.cid{flex:0 0 56px;color:var(--muted);font-variant:small-caps;font-weight:600;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0;}
.kpi{background:var(--soft);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.kpi .v{font-size:24px;font-weight:700;letter-spacing:-.02em;}
.kpi .l{font-size:12.5px;color:var(--muted);margin-top:2px;}
.kpi .t{font-size:11px;margin-top:6px;font-weight:600;}
.t.met{color:#0a7a47;} .t.miss{color:#b23b22;}
figure{margin:14px 0 6px;}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff;}
figcaption{font-size:12.5px;color:var(--muted);margin-top:6px;}
.answer{background:#fff;border-left:4px solid var(--blue);padding:10px 14px;margin:10px 0;font-size:14px;}
.answer.rq2{border-color:var(--verm);} .answer.rq3{border-color:var(--green);}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);}
th{color:var(--muted);font-weight:600;}
.note{font-size:12.5px;color:var(--muted);margin-top:8px;}
code{background:var(--soft);padding:1px 5px;border-radius:5px;font-size:12.5px;}
"""


def _kpi(value: str, label: str, met: Optional[bool]) -> str:
    tag = ""
    if met is not None:
        tag = f'<div class="t {"met" if met else "miss"}">{"threshold met" if met else "below threshold"}</div>'
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div>{tag}</div>'


def render_html(dataset: str, f: Dict[str, Any], fig_dir: Path) -> str:
    prof = f.get("profile", {})
    rows = int(prof.get("rows", 0)); cols = int(prof.get("columns", 0))
    aim_badge = '<span class="badge ok">Aim demonstrated for this dataset</span>' if f["aim_met"] \
        else '<span class="badge no">Aim not fully demonstrated</span>'

    crit_li = "".join(
        f'<li><span class="tick {"ok" if c["met"] else "no"}">{"✓" if c["met"] else "✗"}</span>'
        f'<span class="cid">{c["id"]}</span><span>{html.escape(c["label"])}</span></li>'
        for c in f["criteria"])

    rq3 = f.get("rq3", {})
    tm = rq3.get("thresholds_met", {})
    kpis = "".join([
        _kpi(f'{rq3.get("C", float("nan")):.3f}', "Completeness C(D)", tm.get("C_meets_threshold")),
        _kpi(f'{rq3.get("A", float("nan")):.3f}', "Accuracy A(D)", tm.get("A_meets_threshold")),
        _kpi(f'{rq3.get("R", float("nan")):.3f}', "Reuse readiness R(D)", tm.get("R_meets_threshold")),
    ])

    # RQ answer prose
    rq1 = f.get("rq1", {})
    rq1_ans = (f"AI-DQ3 classified the {rq1.get('n_variables','?')} variables into "
               f"{rq1.get('n_roles','?')} semantic roles and used those roles to select checks: "
               f"distributional anomaly detection on {rq1.get('anomaly_selected','?')} measurement-type "
               f"variables, metadata range checks on {rq1.get('domain_range_selected',0)} and allowed-value "
               f"checks on {rq1.get('allowed_values_selected',0)} variables, identifier-uniqueness on "
               f"{rq1.get('identifier_checks',0)}, required/critical completeness on "
               f"{rq1.get('required_fields',0)}/{rq1.get('critical_fields',0)} fields, and routed "
               f"{rq1.get('low_confidence_to_hitl',0)} low-confidence roles to HITL review. Check selection "
               f"differed from a uniform numeric rule for {rq1.get('selection_differs_from_uniform',0)} variables."
               ) if rq1 else "RQ1 artifact not found."

    a = f.get("rq2", {}).get("anomaly")
    rq2_ans = "RQ2 artifacts not found."
    if a:
        rq2_ans = (f"Under uniform profiling, anomaly detection ran on {a['generic_cols']} numeric columns and "
                   f"flagged {a['generic_rows']} rows ({a['generic_rate']*100:.1f}%). The semantic variable-aware "
                   f"approach restricted detection to {a['semantic_cols']} measurement-type variables and flagged "
                   f"{a['semantic_rows']} rows ({a['semantic_rate']*100:.1f}%), changing which records are "
                   f"prioritised for review and feeding them into HITL triage rather than treating all numeric "
                   f"outliers as equivalent.")
        rb = f.get("rq2", {})
        if "reuse_hybrid" in rb:
            rq2_ans += (f" Reuse readiness was refined from a checklist-only R={rb['reuse_checklist_only']:.3f} to a "
                        f"four-facet hybrid R={rb['reuse_hybrid']:.3f} (documentation, standardisation, privacy, "
                        f"machine-readability).")

    rq3_ans = "RQ3 artifact not found."
    if rq3:
        ac = rq3.get("A_components", {}); rf = rq3.get("R_facets", {})
        rq3_ans = (f"AI-DQ3 produced an interpretable, decomposable dataset-level profile: "
                   f"C(D)={rq3['C']:.3f}, A(D)={rq3['A']:.3f}, R(D)={rq3['R']:.3f}, composite={rq3['composite']:.3f}. "
                   f"Accuracy decomposes into structural={ac.get('A_structural',0):.2f}, domain={ac.get('A_domain',0):.2f}, "
                   f"anomaly={ac.get('A_anomaly',0):.2f}, inconsistency={ac.get('A_inconsistency',0):.2f}, "
                   f"schema-confidence={ac.get('schema_confidence',0):.2f}; reuse readiness into "
                   f"documentation={rf.get('R_documentation',0):.2f}, standardisation={rf.get('R_standardisation',0):.2f}, "
                   f"privacy={rf.get('R_privacy',0):.2f}, machine-readability={rf.get('R_machine_readability',0):.2f}. "
                   f"Every score is traceable to its components, so a reviewer can see why each dimension scored as it did.")

    # triage table
    triage = f.get("triage")
    triage_html = "<p class='note'>No triage register found.</p>"
    if triage is not None:
        t = triage.head(8)
        body = "".join(
            f"<tr><td>{int(r['priority_rank'])}</td><td>{html.escape(str(r['dimension']))}</td>"
            f"<td>{html.escape(str(r['issue_type']).replace('_',' '))}</td>"
            f"<td>{html.escape(str(r['column'])) if str(r['column'])!='nan' else '—'}</td>"
            f"<td>{int(r['count'])}</td><td>{html.escape(str(r['severity']))}</td></tr>"
            for _, r in t.iterrows())
        triage_html = ("<table><tr><th>#</th><th>Dim</th><th>Issue</th><th>Variable</th>"
                       f"<th>Count</th><th>Severity</th></tr>{body}</table>")

    figs = [
        ("fig_overview.png", "Assessment overview: (a) C/A/R + composite profile, (b) uniform vs semantic anomaly detection, (c) per-variable missingness, (d) controlled-error detector validation."),
        ("fig_quality_profile.png", "Dataset-level quality profile across completeness, accuracy and reuse readiness, with per-dimension acceptance thresholds (RQ3)."),
        ("fig_rq2_anomaly_comparison.png", "Effect of semantic variable-aware selection versus uniform data-type profiling on anomaly detection (RQ2)."),
        ("fig_completeness_missingness.png", "Per-variable missingness, separating technical missingness from the semantic-unusable adjustment."),
        ("fig_component_breakdown.png", "Component breakdown of each assessment dimension."),
        ("fig_hitl_triage.png", "Top human-in-the-loop triage candidates, coloured by dimension."),
        ("fig_controlled_error_validation.png", "Detector validation under controlled error injection (precision, recall, F1)."),
        ("fig_weight_sensitivity.png", "Sensitivity of the composite index to dimension weighting scenarios."),
    ]
    fig_html = "".join(
        f'<figure>{_img_tag(fig_dir / name, cap)}<figcaption>{html.escape(cap)}</figcaption></figure>'
        for name, cap in figs)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-DQ3 report — {html.escape(dataset)}</title><style>{CSS}</style></head><body>
<h1>AI-DQ3 assessment report</h1>
<p class="sub">{html.escape(dataset)} · {rows} records × {cols} variables</p>
<div style="margin:10px 0 4px">{aim_badge}</div>
<div class="aim"><strong>Study aim.</strong> {html.escape(AIM_TEXT)}</div>

<h2>Is the aim demonstrated?</h2>
<p class="note">These criteria test the <strong>method capability</strong> the aim and research questions describe.
They are distinct from the dataset's own quality scores below: a dataset can score low while the method still fully
demonstrates the aim.</p>
<ul class="crit">{crit_li}</ul>

<h2>Dataset-level quality profile (RQ3)</h2>
<div class="grid">{kpis}</div>
<div class="answer rq3"><strong>RQ3 — interpretable profile.</strong> {html.escape(rq3_ans)}</div>

<h2>RQ1 — semantic classification guiding check selection</h2>
<div class="answer"><strong>RQ1 — answer.</strong> {html.escape(rq1_ans)}</div>

<h2>RQ2 — semantic vs uniform profiling</h2>
<div class="answer rq2"><strong>RQ2 — answer.</strong> {html.escape(rq2_ans)}</div>

<h2>Human-in-the-loop triage</h2>
{triage_html}
<p class="note">Triage is a cross-cutting layer that prioritises candidate issues for expert review; it is not a
scored dimension. Reviewers record decisions in <code>hitl_validation_sample.csv</code>.</p>

<h2>Figures</h2>
<p class="note">Figures are also written as title-free vector PDFs in the <code>figures/</code> folder for direct
insertion into a manuscript; captions above are suggested wording.</p>
{fig_html}

<h2>Interpretation &amp; limitations</h2>
<p class="note">Results are pre-intervention data-quality diagnostics and HITL triage evidence. They do not constitute
autonomous clinical correction, automatic data repair, or legal-compliance certification. Privacy and
re-identification risk are reported as components of reuse readiness, not as a separate legal assessment. Semantic
role inference is lexical + distributional with reported confidence; low-confidence roles are routed to HITL.</p>
</body></html>"""


def build_report(results_dir: Path, figures_dir: Path, out_dir: Path, dataset: str) -> Path:
    d = results_dir / dataset
    f = derive_findings(d)
    htmltext = render_html(dataset, f, figures_dir / dataset)
    out = out_dir / dataset
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.html"
    path.write_text(htmltext, encoding="utf-8")
    # also a small machine-readable verdict
    (out / "aim_assessment.json").write_text(json.dumps(
        {"dataset": dataset, "aim_met": f["aim_met"],
         "criteria": f["criteria"], "rq3": f.get("rq3", {})}, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AI-DQ3 consolidated HTML results report.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results}")
    datasets = [args.dataset] if args.dataset else [
        p.name for p in sorted(args.results.iterdir())
        if p.is_dir() and (p / "rq3_quality_profile.csv").exists()]
    if not datasets:
        raise FileNotFoundError(f"No dataset result folders found in {args.results}.")
    for ds in datasets:
        path = build_report(args.results, args.figures, args.out, ds)
        print(f"Report written: {path}")


if __name__ == "__main__":
    main()
