# 🧠 Cecilia-MIS (Market Intelligence System)

**Goal:** Identify and analyze emerging market trends for digital product creation with verifiable Google Trends evidence and deterministic opportunity scoring.

## 🔍 Overview
Cecilia-MIS ships two coordinated components:

1. **Google Trends Collector** – uses the official pytrends client to pull interest-over-time signals, summarize volume, 90‑day growth, and current value, and export normalized CSVs.
2. **NT-LEWD Scorer** – enriches each trend row with buyer intent and desperation heuristics, producing a 0‑100 opportunity score and preview table for analyst review.

The repository demonstrates responsible API usage (retry handling, deterministic aggregation, and CSV provenance) to align with Google Trends alpha onboarding requirements.

## 🧱 Tech Stack
- Python 3.10+
- pandas, pytrends, tabulate, PyYAML
- pytest for unit and integration coverage

## 🚀 Collecting Google Trends data

Collect fresh signals directly from Google Trends:

```bash
python main.py \
  --keywords "ai tutor" "student planner" \
  --keywords-file data/seed_keywords.txt \
  --geo US \
  --days 90 \
  --output-dir out \
  --autoscore \
  --score-config config/lewd.yml
```

- `--keywords` supply ad-hoc topics from the CLI.
- `--keywords-file` reads a newline-delimited list (comments starting with `#` are ignored).
- `--autoscore` runs the NT-LEWD scorer immediately after collection.

You can still process an existing CSV export by passing `--input out/trends_20251104.csv`.

## 🎯 Scoring only

Run NT-LEWD scoring against a trends CSV exported by the collector:

```bash
python lewd_scoring.py --input out/trends_20251104.csv --top 15
```

Provide optional configuration and enrichment data for more context-aware scores:

```bash
python lewd_scoring.py \
  --input out/trends_20251104.csv \
  --config config/lewd.yml \
  --pain data/pain_signals.csv \
  --buyers data/who_pays_map.csv
```

## ✅ Testing

```bash
pytest -q
```

## 📤 Publishing updates

After committing local changes, push them to your remote repository so collaborators can access the latest code:

```bash
git push origin <branch-name>
```

Replace `<branch-name>` with the branch you are working on (for example, `main` or `work`). If you created a new branch locally, run `git push -u origin <branch-name>` the first time so future pushes do not require extra flags.

## ⚙️ License
MIT License © 2025 r1cegod
