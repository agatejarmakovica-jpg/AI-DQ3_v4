"""AI-DQ3 — one-step runner for Google Colab / Jupyter.

Usage in Colab:
    !unzip -o ai_dq3_v2_colab_ready.zip
    %cd /content/ai_dq3_v2
    %run colab_bootstrap.py

What it does, in order:
    1. installs dependencies (in the SAME interpreter the notebook uses);
    2. if running in Colab and data/ is empty, opens an upload dialog;
    3. sorts uploaded files (data -> data/, metadata -> metadata/);
    4. runs the assessment IN-PROCESS so any error is shown, not hidden;
    5. builds figures + the HTML report and previews them inline.

Re-run safely. To assess more datasets, drop more files into data/ and re-run.
"""

import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ["data", "metadata", "results", "figures"]:
    (ROOT / sub).mkdir(exist_ok=True)

DATA_EXT = {".csv", ".xlsx", ".xls", ".tsv"}
META_EXT = {".json", ".txt", ".md", ".pdf", ".docx"}

# A tabular file whose name marks it as a codebook / data dictionary / readme is
# documentation, not a dataset to assess: route it to metadata/ so it raises the
# reuse-readiness documentation score (kept in sync with pipeline._DOC_NAME_HINTS).
_DOC_NAME_HINTS = ("readme", "codebook", "code_book", "code book", "dictionary",
                   "data_dictionary", "datadictionary", "description", "documentation", "notes")


def _install_deps() -> None:
    print("Installing dependencies (can take ~30 s the first time)...")
    base = [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")]
    r = subprocess.run(base, capture_output=True, text=True)
    if r.returncode != 0 and "externally-managed-environment" in (r.stderr or ""):
        subprocess.run(base + ["--break-system-packages"], check=False)


def _list_data() -> list:
    return [p for p in (ROOT / "data").iterdir() if p.is_file() and p.suffix.lower() in DATA_EXT]


def _route_file(src: Path) -> None:
    """Place one file into data/ or metadata/ by extension. Recurse into ZIPs."""
    ext = src.suffix.lower()
    if ext == ".zip":
        import zipfile, tempfile
        try:
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(str(src)) as zf:
                    zf.extractall(td)
                for inner in Path(td).rglob("*"):
                    if inner.is_file() and not inner.name.startswith(".") and "__MACOSX" not in str(inner):
                        _route_file(inner)
            print(f"  unpacked {src.name}")
        except Exception as exc:
            print(f"  (could not unzip {src.name}: {exc})")
        return
    try:
        is_doc_named = any(h in src.stem.lower() for h in _DOC_NAME_HINTS)
        if ext in DATA_EXT and is_doc_named:
            shutil.copy(str(src), str(ROOT / "metadata" / src.name)); print(f"  -> metadata/{src.name} (documentation)")
        elif ext in DATA_EXT:
            shutil.copy(str(src), str(ROOT / "data" / src.name)); print(f"  -> data/{src.name}")
        elif ext in META_EXT:
            shutil.copy(str(src), str(ROOT / "metadata" / src.name)); print(f"  -> metadata/{src.name}")
        else:
            print(f"  (ignored {src.name}: not a recognised data/metadata type)")
    except Exception as exc:
        print(f"  (could not place {src.name}: {exc})")


def _maybe_upload() -> None:
    if _list_data():
        print(f"Found {len(_list_data())} dataset(s) already in data/. Skipping upload.")
        return
    try:
        from google.colab import files  # type: ignore
    except Exception:
        print("Not running in Colab. Put your file(s) — or a ZIP with dataset + codebook + "
              "metadata — in the data/ folder (or unzip into data/ and metadata/) and re-run.")
        return
    print("data/ is empty. Upload your dataset, or a ZIP containing the dataset plus any "
          "codebook / readme / metadata files. CSV/XLSX, JSON, TXT/MD, PDF, DOCX and ZIP are accepted.")
    uploaded = files.upload()
    for name in list(uploaded.keys()):
        local = Path(name)
        _route_file(local)
        try:
            if local.exists():
                local.unlink()
        except Exception:
            pass


def run() -> None:
    _install_deps()
    _maybe_upload()

    data_files = _list_data()
    if not data_files:
        print("\n" + "=" * 64)
        print("NO DATA TO ASSESS.")
        print(f"Put at least one .csv or .xlsx file in:\n   {ROOT / 'data'}")
        print("In Colab: use the left file browser, or re-run this cell and use")
        print("the upload dialog. Then run again.")
        print("=" * 64)
        return

    print(f"\nAssessing {len(data_files)} dataset(s): {', '.join(p.name for p in data_files)}")

    sys.path.insert(0, str(ROOT))
    import importlib
    import pipeline as pl, figures as fg, report as rp
    importlib.reload(pl); importlib.reload(fg); importlib.reload(rp)

    config_path = ROOT / "config.yaml"
    print("\n--- Running assessment ---")
    summary = pl.run_pipeline(config_path, verbose=True)

    if summary["failed"]:
        print("\nSome datasets failed. First traceback for debugging:")
        print(summary["failed"][0]["traceback"])
    if not summary["succeeded"]:
        print("No datasets were assessed successfully. See the error(s) above.")
        return

    print("\n--- Building figures ---")
    try:
        fg.SHOW_TITLES = False
        for ds in summary["succeeded"]:
            fg.build_for_dataset(ROOT / "results", ROOT / "figures", Path(ds).stem)
    except Exception as exc:
        print(f"Figure generation issue: {exc}")

    print("\n--- Building report ---")
    try:
        for ds in summary["succeeded"]:
            rp.build_report(ROOT / "results", ROOT / "figures", ROOT / "results", Path(ds).stem)
    except Exception as exc:
        print(f"Report generation issue: {exc}")

    try:
        from IPython.display import Image, display, Markdown  # type: ignore
        for ds in summary["succeeded"]:
            stem = Path(ds).stem
            rep = ROOT / "results" / stem / "report.html"
            ov = ROOT / "figures" / stem / "fig_overview.png"
            display(Markdown(f"### {stem}"))
            if rep.exists():
                display(Markdown(
                    f"Full report: open `results/{stem}/report.html`, or in a new cell:\n"
                    f"```python\nfrom IPython.display import HTML\nHTML(open(r'{rep}').read())\n```"))
            if ov.exists():
                display(Image(filename=str(ov)))
    except Exception:
        pass

    print("\nDone. Open results/<dataset>/report.html for the full review.")


run()
