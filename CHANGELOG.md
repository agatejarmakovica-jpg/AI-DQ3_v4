# AI-DQ3 — Changelog

All changes are confined to two files: `pipeline.py` and `colab_bootstrap.py`.
No scoring weights, thresholds, formulas, figure logic, or configuration were
altered: `config.yaml`, `figures.py`, `report.py`, `README.md` and
`requirements.txt` are unchanged. The composite formula (0.40·A + 0.30·C +
0.30·R), the three dimensions, the four reuse facets and their weights, and the
HITL triage layer are all as originally specified.

The edits address dataset *ingestion* and *documentation attachment* only — i.e.
they let the pipeline read real-world files correctly and stop a scaffolding
file from being mistaken for dataset documentation. They do not change how a
correctly-ingested dataset is scored.

---

## v3 — robust ingestion + documentation-attachment integrity

### 1. `pipeline.load_dataset` — dataset-agnostic delimited-text reader
**Problem.** The original reader called `pd.read_csv(path)` (comma, UTF-8) and
only tried `sep=";"` when that produced exactly one column. A European CSV
(`;`-separated, decimal comma, Latin-1, e.g. `µg`) raised `UnicodeDecodeError`,
the Latin-1 fallback still used a comma separator and raised
`ParserError: Expected N fields, saw N+1`, and the dataset failed to load
entirely. On the first real dataset this caused **3/3 ingestion failures**.

**Fix.** Encoding is auto-detected (UTF-8, else Latin-1). The field delimiter is
chosen from `{",", ";", tab, "|"}` as the one yielding the widest, most
row-consistent table. A European decimal comma is detected from the raw text
(a digit-comma-digit pattern when the delimiter is not a comma) so numeric
columns parse as numbers, not strings. Nothing is hard-coded to any file; the
choice is made from the data.

**Effect.** Comma-CSV, `;`-CSV (incl. decimal comma / Latin-1) and tab files all
load. The European fibromyalgia dataset went from 3/3 failures to fully assessed
(FM 17×44, controls 24×20).

### 2. `pipeline._extract_text_from_file` — `.csv`/`.tsv` documentation branch
**Problem.** The function's docstring promised ".csv-as-text" support, but no
`.csv` branch existed, so a codebook supplied as a CSV produced no documentation
text even if it reached `metadata/`.

**Fix.** Added a `.csv`/`.tsv` text branch (UTF-8, Latin-1 fallback) so a
delimited-file codebook / data dictionary feeds the documentation facet, as the
design intended.

### 3. `pipeline.find_related_metadata` — accept `.csv`/`.tsv` as documentation
**Fix.** Added `.csv`/`.tsv` to the accepted documentation extensions so a
codebook routed to `metadata/` is actually read.

### 4. `pipeline.find_related_metadata` — exclude tool scaffolding (`_TOOL_SCAFFOLDING`)
**Problem.** For a single-dataset run (`n_datasets == 1`), the function attaches
every file in `metadata/` "because loose files clearly belong to it". But
`metadata/` always ships the tool's own `METADATA_GUIDE.md` and
`metadata_template.json`. These were attached as if they were documentation *of
the dataset*: the guide's text (≈6024 chars) fed the documentation TF-IDF
similarity and the template's empty placeholder fields were merged as metadata.
A dataset with **no documentation at all** therefore received a spurious
documentation score (e.g. diabetes `documentation = 0.062`). Multi-dataset runs
were unaffected (the guide has no filename overlap), so documentation scores
were **not comparable** between single- and multi-dataset runs.

**Fix.** A `_TOOL_SCAFFOLDING = {"metadata_guide.md", "metadata_template.json"}`
guard skips those exact shipped filenames. Restricted to the tool's own
filenames so a user's genuine `ReadMe.pdf` / codebook is never excluded.

**Effect (diabetes, no accompanying document).**

| Quantity | Before | After |
|---|---|---|
| attached doc text | 6024 chars (the tool guide) | 0 chars |
| `documentation` facet | 0.062 | 0.000 |
| Reuse readiness R | 0.275 | 0.250 |
| Composite | 0.730 | 0.723 |

Datasets with real documentation are unchanged: the fibromyalgia codebook and
the formulation `ReadMe.pdf` still attach (verified doc text ≈3633 chars).

### 5. `colab_bootstrap._route_file` — route documentation-named tabular files to `metadata/`
**Problem.** Routing was purely by extension, so a `.csv` codebook (e.g.
`Codebook_FM .csv`) landed in `data/` and was assessed as if it were a dataset,
while contributing nothing to the documentation score.

**Fix.** Added `_DOC_NAME_HINTS` (kept in sync with `pipeline._DOC_NAME_HINTS`).
A tabular file whose name marks it as a codebook / data dictionary / readme is
routed to `metadata/` (documentation) instead of `data/`.

---

## Change size
- `pipeline.py`: +73 / −10 lines (4 edits)
- `colab_bootstrap.py`: +10 / −1 line (1 edit)
- `config.yaml`, `figures.py`, `report.py`, `README.md`, `requirements.txt`: unchanged

See `changes.diff` (unified diff against the pre-change version) for the exact edits.

---

## v4 — sentence-transformers un cross-encoder obligāti ieslēgti

### Izmaiņas

Tikai divi faili mainīti: `config.yaml` un `requirements-semantic.txt`.
`pipeline.py` nav skarts — loģika jau atbalstīja abas metodes caur `enabled` karodziņu.

#### 1. `config.yaml` — `semantic_role_inference.embedding.enabled`: `false` → `true`

Sentence-transformers bi-enkoderis tagad vienmēr aktīvs lomu piešķiršanai.
Katra kolonnas nosaukums + paraugvērtības + (ja pieejams) metadata apraksts tiek enkodēts
un salīdzināts ar lomu prototipiem pēc kosinusa līdzības. Tas aizstāj vai papildina
regex modeļu heiristiku ar nozīmes balstītu atpazīšanu (sinonīmi, neredzēti nosaukumi).
Fallback uz leksikālajām heiristikām ja bibliotēka nav instalēta vai modelis nav
pieejams (pipeline nekad neapstājas).

#### 2. `config.yaml` — `reuse_readiness.semantic_similarity.cross_encoder.enabled`: `false` → `true`

Cross-encoder (ms-marco-MiniLM-L-6-v2) tagad aktīvs dokumentācijas relevances
novērtēšanai R dimensijā. Tas ir precīzāks par bi-enkoderu kosinusu jo vērtē
(dokumentācija, reuse_readiness_prompts) pārus tieši. Prioritātes secība saglabāta:
cross-encoder → bi-enkoderu kosinuss → TF-IDF (katra nākamā metode ir fallback).

#### 3. `requirements-semantic.txt` — precizēts komentārs

Skaidri norādīts ka `sentence-transformers>=2.2` iekļauj abus nepieciešamos modeļus.

### Instalēšana

```bash
pip install -r requirements-semantic.txt
```

Tas ielādē `sentence-transformers` pakotni, kas lejupielādē abus modeļus
(`all-MiniLM-L6-v2` un `ms-marco-MiniLM-L-6-v2`) automātiski no HuggingFace
pie pirmās palaišanas.

### Ietekme uz rezultātiem

| Komponents | Pirms (v3) | Pēc (v4) |
|---|---|---|
| `semantic_inference_method` profilā | `lexical_distributional_heuristic` | `embedding:sentence-transformers/all-MiniLM-L6-v2` |
| `text_method` reuse detaļās | `tfidf_cosine_similarity` | `cross_encoder_reranking` |
| `pipeline.py` | neskarti | neskarti |
| Svari, formulas, C/A/R dimensijas | neskarti | neskarti |
