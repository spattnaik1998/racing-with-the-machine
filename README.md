# Racing With the Machine: code

*Racing With the Machine: AI Exposure, Skill, and the Future of the Professions in the Second Machine Age*
Sarthak Pattnaik, Eugene Pinsky, Kathleen Park — Boston University

## Setup (once)

```
cd racing-with-the-machine
python -m venv .venv && .venv\Scripts\activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Copy `.env.example` to `.env` (in this folder or its parent) and add your keys; it is found automatically. It needs `BEA_API_KEY` and `FRED_API_KEY`. Both `KEY: value` and `KEY=value` lines work. The notebooks never print keys, but BEA echoes the key inside its responses, so the cached files in `data/raw/api/` contain it; `.gitignore` keeps that folder out of the repository.

## Run order

| Notebook | What it does | Main outputs |
|---|---|---|
| `01_fetch_data` | Downloads OEWS 2019–2025, O*NET, Eloundou exposure, BEA, Census BTOS, FRED; caches in `data/raw/` | `data/interim/*` |
| `02_build_panels` | Occupation × year panel; skill indices (social, synchronization, ideation); industry labour-share panel | `data/final/occ_panel.csv`, `industry_panel.csv` |
| `03_H1_exposure_by_skill` | H1: exposure by Job Zone and wage decile | Tables 1–3, Fig. 1 |
| `04_H2_H4_event_studies` | H2 employment, H3 protective skills, H4 wage dispersion; event studies and pooled DiD | Table 4, Figs. 2–5 |
| `05_H5_industry_labor_share` | H5: labour share vs pre-2008 IP intensity; wild-cluster bootstrap; AI use by sector | Table 5, Figs. 6–7 |
| `06_robustness_and_paper_numbers` | Robustness checks and placebo; verdict table; every number for the paper | Table 6–7, Fig. 8, `output/numbers_for_paper.json` |

Run them in order: Kernel → Restart & Run All, one notebook at a time. The first run of 01 takes a few minutes (about 150 MB of downloads); later runs use the cache.

## If something fails

Each notebook ends with a short check cell. Send the error or that cell's output. Places most likely to need a tweak with live data:

- **OEWS May 2025** may not be published under the expected file name yet; 01 skips missing years and the analysis adapts to the years present.
- **BEA tables** are looked up by name. If BEA changed the layout, 02 stops with a printout of the rows it found.
- **Industry name matching** between the two BEA tables: check `data/interim/ip_industry_mapping.csv`, and add any unmatched private industry to `MANUAL_FA_MATCH` in 02.
- **Census BTOS** sector file layout varies; it is optional and H5 runs without it.

## Methods notes for the paper

- Occupation models absorb occupation and year (or major-group × year) fixed effects by demeaning, with SEs clustered by occupation. This matches `reghdfe` in Stata.
- Exposure is the Eloundou et al. (2024) β score, z-scored across occupations; robustness uses the human rating and the α and γ scores.
- Occupations with a top-coded 90th-percentile wage in any year are excluded from the P90/P10 models.
- OEWS pools three years of survey panels; BLS cautions against reading it as a time series. Report results as divergence, not causal effects.
