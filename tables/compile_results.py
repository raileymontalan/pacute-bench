import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_colwidth", 40)

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results-202605"
TABLES_DIR = REPO_ROOT / "tables"
TABLES_DIR.mkdir(exist_ok=True)

print("Repo root:", REPO_ROOT.resolve())

# ── Model configs ─────────────────────────────────────────────────────────────

def load_model_configs() -> dict[str, str]:
    cfg_map = {
        REPO_ROOT / "configs" / "models_pt.yaml":         "PT",
        REPO_ROOT / "configs" / "models_it.yaml":         "IT",
        REPO_ROOT / "configs" / "models_commercial.yaml": "Commercial",
    }
    out: dict[str, str] = {}
    for path, group in cfg_map.items():
        if not path.exists():
            print(f"  WARNING: config not found: {path}")
            continue
        data = yaml.safe_load(path.read_text())
        for name in data.get("models", {}):
            out[name] = group
    return out

MODEL_GROUPS = load_model_configs()
print(f"Loaded {len(MODEL_GROUPS)} model entries")
for g in ["PT", "IT", "Commercial"]:
    names = [n for n, grp in MODEL_GROUPS.items() if grp == g]
    print(f"  {g}: {len(names)} models")

# ── Benchmark registry ────────────────────────────────────────────────────────

BENCHMARK_GROUP = {
    "pacute-composition-mcq":     ("pacute", "composition",     "MCQ"),
    "pacute-composition-gen":     ("pacute", "composition",     "Gen"),
    "pacute-manipulation-mcq":    ("pacute", "manipulation",    "MCQ"),
    "pacute-manipulation-gen":    ("pacute", "manipulation",    "Gen"),
    "pacute-morphological-extraction-mcq":      ("pacute", "morphological-extraction",      "MCQ"),
    "pacute-morphological-extraction-gen":      ("pacute", "morphological-extraction",      "Gen"),
    "pacute-morphological-production-mcq":      ("pacute", "morphological-production",      "MCQ"),
    "pacute-morphological-production-gen":      ("pacute", "morphological-production",      "Gen"),
    "pacute-syllabification-mcq": ("pacute", "syllabification", "MCQ"),
    "pacute-syllabification-gen": ("pacute", "syllabification", "Gen"),
    "hierarchical-mcq":           ("other",  "hierarchical",    "MCQ"),
    "hierarchical-gen":           ("other",  "hierarchical",    "Gen"),
    "langgame-mcq":               ("other",  "langgame",        "MCQ"),
    "langgame-gen":               ("other",  "langgame",        "Gen"),
    "multi-digit-addition-mcq":   ("other",  "multi-digit-add", "MCQ"),
    "multi-digit-addition-gen":   ("other",  "multi-digit-add", "Gen"),
    "cute-gen":                   ("other",  "cute",            "Gen"),
}

BENCH_ABBR = {
    "pacute-composition-mcq":     "Comp-MCQ",
    "pacute-composition-gen":     "Comp-Gen",
    "pacute-manipulation-mcq":    "Manip-MCQ",
    "pacute-manipulation-gen":    "Manip-Gen",
    "pacute-morphological-extraction-mcq":      "MExt-MCQ",
    "pacute-morphological-extraction-gen":      "MExt-Gen",
    "pacute-morphological-production-mcq":      "MProd-MCQ",
    "pacute-morphological-production-gen":      "MProd-Gen",
    "pacute-syllabification-mcq": "Syll-MCQ",
    "pacute-syllabification-gen": "Syll-Gen",
    "hierarchical-mcq":           "Hier-MCQ",
    "hierarchical-gen":           "Hier-Gen",
    "langgame-mcq":               "LGame-MCQ",
    "langgame-gen":               "LGame-Gen",
    "multi-digit-addition-mcq":   "MDA-MCQ",
    "multi-digit-addition-gen":   "MDA-Gen",
    "cute-gen":                   "CUTE-Gen",
}

BENCH_FULLNAME = {
    "MExt":         "PACUTE Morphological Extraction",
    "MProd":        "PACUTE Morphological Production",
    "Comp":         "PACUTE Composition",
    "Manip":        "PACUTE Manipulation",
    "Syll":         "PACUTE Syllabification",
    "Hier":         "Hierarchical",
    "LGame":        "LangGame",
    "MDA":          "Multi-Digit Addition",
    "CUTE":         "CUTE",
}

GROUPS_ORDER = ["PT", "IT", "Commercial", "Unknown"]

# ── Load results ──────────────────────────────────────────────────────────────

def latest_result_file(model_dir: Path) -> Path | None:
    files = sorted(model_dir.glob("evaluation_results_*.json"), reverse=True)
    for f in files:
        if json.loads(f.read_text()).get("benchmarks"):
            return f
    return None


rows = []

for model_dir in sorted(RESULTS_DIR.iterdir()):
    if not model_dir.is_dir() or model_dir.name == "tables":
        continue
    result_file = latest_result_file(model_dir)
    if result_file is None:
        continue

    model_key = model_dir.name
    group     = MODEL_GROUPS.get(model_key, "Unknown")
    data      = json.loads(result_file.read_text())
    model_type = data.get("model_type", "?")
    thinking   = data.get("thinking", False)
    hf_model_name = data.get("hf_model_name", model_key)
    model_name = f"{hf_model_name} (thinking)" if thinking else hf_model_name

    for bench_name, bench_data in data.get("benchmarks", {}).items():
        if bench_name not in BENCHMARK_GROUP:
            continue
        category, subcategory_top, task_type = BENCHMARK_GROUP[bench_name]

        base = {
            "model":      model_name,
            "group":      group,
            "model_type": model_type,
            "thinking":   thinking,
            "benchmark":  bench_name,
            "category":   category,
            "subcategory": subcategory_top,
            "task_type":  task_type,
            "split":      "overall",
            "n":          bench_data.get("num_samples", 0),
        }
        if task_type == "MCQ":
            base["norm_acc"] = bench_data.get("normalized_accuracy", np.nan)
            base["f1"]       = bench_data.get("f1_score", np.nan)
            base["em"]       = np.nan
            base["cm"]       = np.nan
        else:
            base["norm_acc"] = np.nan
            base["f1"]       = np.nan
            base["em"]       = bench_data.get("exact_match", np.nan)
            base["cm"]       = bench_data.get("contains_match", np.nan)
        rows.append(base)

        for sub, sub_data in bench_data.get("by_category", {}).items():
            sub_row = {**base, "split": sub, "n": sub_data.get("num_samples", 0)}
            if task_type == "MCQ":
                sub_row["norm_acc"] = sub_data.get("normalized_accuracy", np.nan)
                sub_row["f1"]       = sub_data.get("f1_score", np.nan)
                sub_row["em"]       = np.nan
                sub_row["cm"]       = np.nan
            else:
                sub_row["norm_acc"] = np.nan
                sub_row["f1"]       = np.nan
                sub_row["em"]       = sub_data.get("exact_match", np.nan)
                sub_row["cm"]       = sub_data.get("contains_match", np.nan)
            rows.append(sub_row)

df = pd.DataFrame(rows)
print(f"Total rows: {len(df)}")
print(f"Models: {df['model'].nunique()}")
print(f"Groups: {df['group'].value_counts().to_dict()}")

overall = df[df["split"] == "overall"].copy()

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(val, pct=True):
    if pd.isna(val):
        return "—"
    return f"{val*100:.1f}" if pct else f"{val:.4f}"


def build_compact(group: str) -> pd.DataFrame:
    sub = overall[overall["group"] == group].copy()
    if sub.empty:
        return pd.DataFrame()

    records = {}
    for _, row in sub.iterrows():
        m  = row["model"]
        b  = row["benchmark"]
        tt = row["task_type"]
        if m not in records:
            records[m] = {"group": group}
        if tt == "MCQ":
            records[m][f"{b} / NormAcc"] = row["norm_acc"]
            records[m][f"{b} / F1"]      = row["f1"]
        else:
            records[m][f"{b} / EM"] = row["em"]
            records[m][f"{b} / CM"] = row["cm"]

    out = pd.DataFrame.from_dict(records, orient="index")
    out.index.name = "model"
    bench_order = [b for b in BENCHMARK_GROUP if b in overall["benchmark"].unique()]
    col_order = ["group"]
    for b in bench_order:
        tt = BENCHMARK_GROUP[b][2]
        for suf in (["NormAcc", "F1"] if tt == "MCQ" else ["EM", "CM"]):
            c = f"{b} / {suf}"
            if c in out.columns:
                col_order.append(c)
    return out[[c for c in col_order if c in out.columns]].sort_index()

# ── Section 4: Per-benchmark summary ─────────────────────────────────────────

for group in GROUPS_ORDER:
    grp_models = sorted(overall[overall["group"] == group]["model"].unique())
    if not grp_models:
        continue
    print(f"\n{'═'*100}")
    print(f"  GROUP: {group}  ({len(grp_models)} models: {', '.join(grp_models)})")
    print(f"{'═'*100}")
    for task in ["MCQ", "Gen"]:
        sub = overall[(overall["group"] == group) & (overall["task_type"] == task)]
        if sub.empty:
            continue
        metric_cols = {"norm_acc": "NormAcc (%)", "f1": "F1 (%)"} if task == "MCQ" else {"em": "EM (%)", "cm": "CM (%)"}
        pivot = sub.pivot_table(
            index=["category", "subcategory", "benchmark"],
            columns="model",
            values=list(metric_cols.keys()),
            aggfunc="first",
        )
        pivot.columns = [f"{metric_cols[m]}  {mdl}" for m, mdl in pivot.columns]
        print(f"\n── {task} ──")
        print(pivot.to_string(float_format=lambda v: fmt(v)))

# ── Section 5: Per-subcategory breakdown ──────────────────────────────────────

detail = df[df["split"] != "overall"].copy()

for bench_name in sorted(df["benchmark"].unique()):
    bench_df = detail[detail["benchmark"] == bench_name]
    if bench_df.empty:
        continue

    task_type    = bench_df["task_type"].iloc[0]
    metric_col   = "norm_acc" if task_type == "MCQ" else "em"
    metric_label = "NormAcc (%)" if task_type == "MCQ" else "EM (%)"

    bench_df = bench_df.copy()
    bench_df["col"] = bench_df["group"] + " / " + bench_df["model"]

    pivot = bench_df.pivot_table(
        index="split",
        columns="col",
        values=metric_col,
        aggfunc="first",
    )

    def col_sort_key(c):
        grp = c.split(" / ")[0]
        return (GROUPS_ORDER.index(grp) if grp in GROUPS_ORDER else 99, c)

    pivot = pivot[sorted(pivot.columns, key=col_sort_key)]

    print(f"\n{'─'*100}")
    print(f"  {bench_name}  [{task_type}]  — {metric_label}")
    print(f"{'─'*100}")
    print(pivot.to_string(float_format=lambda v: fmt(v)))

# ── Section 6: Compact summary table ─────────────────────────────────────────

def _abbrev_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if " / " in col:
            bench, metric = col.rsplit(" / ", 1)
            abbr = BENCH_ABBR.get(bench, bench)
            abbr_base = abbr.replace("-MCQ", "").replace("-Gen", "")
            rename[col] = f"{abbr_base} / {metric}"
    return df.rename(columns=rename)


for group in GROUPS_ORDER:
    ct = build_compact(group)
    if ct.empty:
        continue
    print(f"\n{'═'*120}")
    print(f"  {group}")
    print(f"{'═'*120}")
    numeric = ct.drop(columns=["group"])
    display_ct = _abbrev_columns(numeric)
    print(display_ct.to_string(float_format=lambda v: fmt(v)))

# ── Section 7: LaTeX export ───────────────────────────────────────────────────

LATEX_HEADER = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{adjustbox}
\usepackage{threeparttable}
\usepackage{colortbl}
\usepackage[margin=1in]{geometry}
\begin{document}
"""
LATEX_FOOTER = r"\end{document}" + "\n"


def _escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _cell_color(val: float, vmin: float, vmax: float) -> str | None:
    if math.isnan(val) or vmax == vmin:
        return None
    t = (val - vmin) / (vmax - vmin)
    if t <= 0.5:
        s = t * 2
        r = int(244 + s * (255 - 244))
        g = int(165 + s * (255 - 165))
        b = int(130 + s * (255 - 130))
    else:
        s = (t - 0.5) * 2
        r = int(255 - s * (255 - 146))
        g = int(255 - s * (255 - 197))
        b = int(255 - s * (255 - 222))
    return f"{r:02X}{g:02X}{b:02X}"


def build_latex_table(group: str, task_filter: str | None = None, caption: str | None = None) -> str:
    ct = build_compact(group)
    if ct.empty:
        return ""

    data = ct.drop(columns=["group"]).copy()
    data = data.dropna(axis=1, how="all")

    if task_filter is not None:
        keep = [
            col for col in data.columns
            if task_filter.upper() in BENCH_ABBR.get(col.rsplit(" / ", 1)[0], col).upper()
        ]
        data = data[keep]
    data = data.dropna(axis=1, how="all")
    if data.empty:
        return ""

    col_max_model: dict[str, str] = {}
    col_vmin: dict[str, float] = {}
    col_vmax: dict[str, float] = {}
    for col in data.columns:
        col_numeric = pd.to_numeric(data[col], errors="coerce")
        if col_numeric.notna().any():
            col_max_model[col] = col_numeric.idxmax()
            col_vmin[col] = col_numeric.min()
            col_vmax[col] = col_numeric.max()

    bench_spans: OrderedDict = OrderedDict()
    abbrs_used: set[str] = set()
    for col in data.columns:
        bench, metric = col.rsplit(" / ", 1)
        abbr_full = BENCH_ABBR.get(bench, bench)
        task = "MCQ" if "MCQ" in abbr_full else "Gen"
        abbr_base = abbr_full.replace("-MCQ", "").replace("-Gen", "")
        key = (abbr_base, task)
        bench_spans.setdefault(key, []).append((col, metric))
        abbrs_used.add(abbr_base)

    col_spec = "l" + "r" * len(data.columns)

    header1_parts = [r"\textbf{Model}"]
    header2_parts = [""]
    cmidrules = []
    col_idx = 2
    for (abbr_base, task), cols in bench_spans.items():
        n = len(cols)
        label = rf"\textbf{{{abbr_base}-{task}}}"
        header1_parts.append(rf"\multicolumn{{{n}}}{{c}}{{{label}}}")
        header2_parts += [rf"\textit{{{metric}}}" for _, metric in cols]
        cmidrules.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n

    rows_tex = []
    for model_name, row in data.iterrows():
        cells = [_escape(str(model_name))]
        for col in data.columns:
            val = row[col]
            if pd.isna(val):
                cells.append("—")
            else:
                s = fmt(val)
                if col_max_model.get(col) == model_name:
                    s = rf"\textbf{{{s}}}"
                if col in col_vmin:
                    color = _cell_color(float(val), col_vmin[col], col_vmax[col])
                    if color:
                        s = rf"\cellcolor[HTML]{{{color}}}{s}"
                cells.append(s)
        rows_tex.append(" & ".join(cells) + r" \\")

    note_parts = [
        f"\\textit{{{abbr}}} = {BENCH_FULLNAME[abbr]}"
        for abbr in sorted(abbrs_used)
        if abbr in BENCH_FULLNAME
    ]
    note_parts.append(r"Bold = best in column. Cell colour: \colorbox[HTML]{F4A582}{\strut low} $\to$ white $\to$ \colorbox[HTML]{92C5DE}{high} per column.")

    if caption is None:
        task_desc = f" ({task_filter} benchmarks)" if task_filter else ""
        caption = f"Results for {group} models on PACUTE{task_desc}."
    label_suffix = (f":{task_filter.lower()}" if task_filter else "").replace("-", "")
    table_label = f"tab:results-{group.lower()}{label_suffix}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{table_label}}}",
        r"\begin{threeparttable}",
        rf"\begin{{adjustbox}}{{max width=\textwidth}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header1_parts) + r" \\",
        "".join(cmidrules),
        " & ".join(header2_parts) + r" \\",
        r"\midrule",
        *rows_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\begin{tablenotes}[flushleft]\footnotesize",
        r"\item " + "; ".join(note_parts),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


SPLIT_GROUPS = {"it"}

saved = []
for group in GROUPS_ORDER:
    group_lower = group.lower()

    if group_lower in SPLIT_GROUPS:
        tex_mcq = build_latex_table(group, task_filter="MCQ",
                                    caption=f"Results for {group} models on PACUTE MCQ benchmarks.")
        tex_gen = build_latex_table(group, task_filter="Gen",
                                    caption=f"Results for {group} models on PACUTE GEN benchmarks.")
        if not tex_mcq and not tex_gen:
            continue

        body = f"\\section*{{Results: {group} Models}}\n\n"
        if tex_mcq:
            body += "\\subsection*{MCQ Benchmarks}\n\n" + tex_mcq + "\n"
        if tex_gen:
            body += "\\subsection*{Generation Benchmarks}\n\n" + tex_gen + "\n"

        full_tex = LATEX_HEADER + body + LATEX_FOOTER
        out_path = TABLES_DIR / f"results_{group_lower}.tex"
        out_path.write_text(full_tex)
        saved.append(str(out_path))
        print(f"  {group} (split MCQ + Gen) → {out_path}")
    else:
        tex = build_latex_table(group, caption=f"Results for {group} models on PACUTE benchmarks.")
        if not tex:
            continue

        full_tex = (
            LATEX_HEADER
            + f"\\section*{{Results: {group} Models}}\n\n"
            + tex
            + "\n"
            + LATEX_FOOTER
        )
        out_path = TABLES_DIR / f"results_{group_lower}.tex"
        out_path.write_text(full_tex)
        saved.append(str(out_path))
        print(f"  {group} → {out_path}")

print("\nSaved files:")
for p in saved:
    print(" ", p)
