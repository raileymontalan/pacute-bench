# Human Baselines for PACUTE-Bench

## Purpose

Establish human performance ceilings and inter-annotator agreement (IAA) for every active benchmark. Three annotators each answer the same stratified 10% sample. Human scores are computed with the same scoring pipeline used for models.

---

## Benchmarks in Scope

Affixation (gen + mcq) is **excluded** — deprecated.

| Benchmark | MCQ items | GEN items | Has categories? |
|---|---|---|---|
| `composition` | 950 | 550 | Yes (up to 9 subcats) |
| `manipulation` | 800 | 800 | Yes (8 subcats each) |
| `syllabification` | 200 | 200 | Yes (2 subcats each) |
| `morphological_extraction` | 400 | 400 | Yes (4 subcats each) |
| `morphological_production` | 150 | 150 | 1 subcat |
| `hierarchical` | 600 | 600 | Yes (14 subcats, uneven) |
| `langgame` | 1000 | 1000 | No |
| `multi_digit_addition` | 1000 | 1000 | No |
| `cute` | — | 1400 | Yes (14 task types × 100) |

---

## Sampling Strategy

### Rules

1. **Same sample for all 3 annotators** — required for IAA.
2. **Stratified by `category` + `subcategory`** (or `task_type` for cute). Sample proportionally from each stratum; minimum 1 item per stratum even if stratum < 10 items.
3. **Target: ~10% per benchmark**, per format (MCQ and GEN sampled independently since they are distinct items).
4. For benchmarks with no category field (`langgame`, `multi_digit_addition`): random 10%.
5. Fix a random seed (e.g., `42`) so the sample is reproducible.

### Resulting sample sizes (approximate)

| Benchmark | MCQ sample | GEN sample |
|---|---|---|
| composition | 95 | 55 |
| manipulation | 80 | 80 |
| syllabification | 20 | 20 |
| morphological_extraction | 40 | 40 |
| morphological_production | 15 | 15 |
| hierarchical | 60 | 60 |
| langgame | 100 | 100 |
| multi_digit_addition | 100 | 100 |
| cute (GEN only) | — | 140 |
| **Total per annotator** | **510** | **610** |
| **Grand total per annotator** | | **1,120** |

> 1,120 items per annotator is substantial. If needed, annotators can split work across multiple sessions — the annotation sheets are designed to support resuming.

---

## Annotation Format

### Recommendation: Structured Google Sheets (or Excel)

A simple web annotation app would give the cleanest UX, but the setup and maintenance overhead isn't worth it for a one-time collection with 3 trusted annotators. Google Sheets / Excel is faster to deploy, familiar, easy to audit, and the output can be parsed directly by the scoring script.

**Against a plain CSV:** we need dropdown validation for MCQ (to prevent typos in option labels) and per-row progress tracking. Sheets/Excel handles both natively.

### Sheet layout

One workbook per annotator. Each benchmark gets **two tabs**: one for MCQ, one for GEN (cute gets only a GEN tab).

#### MCQ tab columns

| item_id | system_prompt | question | option_A | option_B | option_C | option_D | correct_label | annotator_answer | notes |
|---|---|---|---|---|---|---|---|---|---|

- `system_prompt`: the instruction string from `configs/evaluation.yaml` (or blank for PT-style benchmarks)
- `question`: the `text_en` field (English prompt shown to model)
- `option_A` … `option_D`: shuffled choices (same shuffle used when generating MCQ items; use `choice1`…`choice4` fields)
- `correct_label`: pre-filled but **hidden** — used only after annotation for scoring. Hide this column during annotation.
- `annotator_answer`: **data entry column** — dropdown restricted to `{A, B, C, D}` (or fewer for syllabification MCQ which has only 2 options)
- `notes`: free-text, optional

#### GEN tab columns

| item_id | system_prompt | question | reference_answer | annotator_answer | notes |
|---|---|---|---|---|---|

- `reference_answer`: pre-filled but **hidden** during annotation
- `annotator_answer`: free-text entry

### What annotators see

Annotators should see exactly what the evaluated models see: the system prompt (if any) and the question. For MCQ, they see the options. They do **not** see the reference answer during annotation.

The generation-setting system prompts (from `configs/evaluation.yaml`) explain the task format. Include these verbatim in the `system_prompt` column so annotators have the same context as models. Annotators should produce their answer in the same format the model is asked to produce (e.g., just the word, just the integer, just the letter A/B/C/D).

---

## Scripts to Build

### 1. `scripts/sample_human_baseline.py`

Reads all active benchmark JSONL files, applies stratified sampling (seed=42), writes:
- `data/human_baseline/sample_mcq.jsonl` — all MCQ samples, fields: `{item_id, benchmark, category, subcategory, system_prompt, question, options, correct_option_label}`
- `data/human_baseline/sample_gen.jsonl` — all GEN samples, fields: `{item_id, benchmark, category, subcategory, system_prompt, question, reference_answer}`

### 2. `scripts/generate_annotation_sheets.py`

Reads `sample_mcq.jsonl` and `sample_gen.jsonl`, produces one Excel workbook per annotator (they're identical at this stage — same items, same order). Output: `data/human_baseline/annotation_<annotator_id>.xlsx`.

- Shuffles option order independently per item (using `item_id` as seed) — same shuffle for all annotators
- Locks the `correct_label` / `reference_answer` columns (hidden + protected)
- Adds dropdown validation on `annotator_answer` for MCQ

### 3. `scripts/score_human_baselines.py`

Reads filled-in workbooks, computes:

**Per annotator:**
- MCQ: `accuracy`, `normalized_accuracy` (same formula as `_by_category_mcq`), per benchmark + per subcategory
- GEN: `exact_match`, `contains_match`, `prefix_match` (same as `_by_category_gen`)

**Aggregated across annotators:**
- Human mean and std per metric
- IAA (see below)

Outputs:
- `results/human_baseline/scores_per_annotator.json`
- `results/human_baseline/iaa.json`
- `results/human_baseline/summary.md` — human-readable table for the paper

---

## Scoring Details

Reuse existing scoring logic from `src/pacute_bench/evaluators/base.py`:

- **MCQ**: compare `annotator_answer` (A/B/C/D) to `correct_option_label`. Compute accuracy and normalized accuracy `(acc * K - 1) / (K - 1)` where K = number of options.
- **GEN**: apply `_normalize_label` to both `annotator_answer` and `reference_answer`, then compute exact_match, contains_match (reference in answer), prefix_match.

No special pipeline needed — the human answer slots directly into the same comparison logic.

---

## Inter-Annotator Agreement

### Why high agreement is expected

All tasks have deterministic, objectively correct answers (character positions, affix forms, arithmetic, etc.). Disagreements should be rare and informative — they flag ambiguous or poorly specified items.

### MCQ: Fleiss' κ

Three annotators, K categorical options (2–4 depending on benchmark). Report:
- Overall κ across all MCQ items
- Per-benchmark κ
- Per-subcategory κ (flag any subcategory with κ < 0.8)

Interpretation targets: κ ≥ 0.8 = strong, 0.6–0.8 = moderate (investigate), < 0.6 = problematic.

### GEN: Percent exact agreement + Fleiss' κ on binary

Because GEN answers are free text, use two complementary metrics:

1. **Percent exact agreement**: fraction of items where all 3 annotators give identical normalized answers. Target ≥ 90%.
2. **Fleiss' κ on binary exact-match**: binarize each annotator's answer as correct (1) or incorrect (0) against the reference, then compute Fleiss' κ across the three binary ratings. This captures whether annotators agree on *which items are hard*, not just whether they agree with each other.

Use `sklearn.metrics.cohen_kappa_score` for pairwise κ and aggregate, or `statsmodels` for multi-rater Fleiss' κ.

### Reporting

For each benchmark, report a table:

```
Benchmark              | Annotators | MCQ acc (mean±std) | GEN exact (mean±std) | MCQ κ | GEN agree%
composition            | 3          | 0.XX ± 0.XX        | 0.XX ± 0.XX          | 0.XX  | XX%
...
```

Items where ≥ 2 annotators disagree with the reference answer should be flagged for manual review — they may indicate annotation errors or benchmark issues.

---

## Workflow

1. Run `scripts/sample_human_baseline.py` → generates locked sample JSONL files.
2. Run `scripts/generate_annotation_sheets.py` → produces 3 identical workbooks.
3. Distribute one workbook per annotator. They fill in `annotator_answer` column only.
4. Collect filled workbooks. Run `scripts/score_human_baselines.py`.
5. Review IAA report; flag low-agreement items for discussion.
6. Human scores are added to the main results table alongside model scores.

---

## Notes

- Annotators should work independently — no discussion until after annotation is complete.
- For Filipino morphology tasks (`morphological_extraction`, `morphological_production`, `syllabification`, `hierarchical`), annotators should be native or near-native Filipino speakers.
- For character-level tasks (`composition`, `manipulation`, `cute`, `langgame`, `multi_digit_addition`), no language background requirement — these are language-independent.
- If an annotator is unsure on a GEN item, they should write their best attempt and add a note. Do not leave `annotator_answer` blank — blanks are scored as incorrect.
