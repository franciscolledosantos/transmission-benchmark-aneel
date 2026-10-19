# transmission-benchmark-aneel

Code and result tables for the manuscript *Model risk in regulatory benchmarking of electricity transmission: evidence from Brazil's 2017–2021 operating-cost database* (submitted to *Utilities Policy*, 2026).

Using the public 2017–2021 database released by the Brazilian electricity regulator (ANEEL) for its periodic review, the study measures how much the operating and maintenance cost recognised for 65 transmission groupings (253 grouping-years) depends on modelling choices. Five DEA specifications (returns to scale, weight restrictions) are compared with a stochastic cost frontier and machine-learning cost norms validated out of sample, with bootstrap intervals, returns-to-scale tests, peer analysis and a second stage on operating conditions (fire activity, Amazon location, network dispersion). Recognised cost ranges from +3% to −65% of actual cost across specifications.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Panel construction, DEA (`dea.py`), bootstrap, SFA, ML cost norms, sensitivity analyses, figures and the manuscript builder |
| `dados/` | INPE active fires by state 2015–2022 (small public file). The ANEEL spreadsheets go here but are not redistributed |
| `resultados/` | Processed panel (`painel_dmu_ano.csv`), efficiency scores by specification, bootstrap, peers, RTS tests, SFA tables, second-stage and sensitivity results |
| `figuras/` | Figs. 1–3 of the manuscript (PNG; the submission uses 300 dpi TIFF rendered by `figuras.py`) |

## Pipeline

1. `scripts/painel.py` — reads the ANEEL spreadsheets (assets, PMSO, availability, groupings, asset age) and INPE fires → `resultados/painel_dmu_ano.csv`.
2. `scripts/analise.py` — DEA under VRS/CRS/NDRS with and without weight restrictions, Simar–Wilson bootstrap → `eficiencias_dmu*.csv`, `crs_bootstrap.csv`, `custo_reconhecido*.csv`, `peers_*.csv`.
3. `scripts/analise_ml.py` — ML cost norms (leave-group-out) → `normas_custo_cv.csv`, `ml_*.csv`.
4. `scripts/extras.py`, `sfa_boot.py`, `sfa_tabela.py` — stochastic frontier, elasticities and cluster bootstrap.
5. `scripts/peers_sens.py`, `rts_test.py`, `rts_test_nirs.py`, `sens_cond_trab.py`, `extras_revisao.py` — revision analyses (peer exclusion, returns-to-scale tests, operating conditions, scale by unit).
6. `scripts/figuras.py` — Figs. 1–3; `scripts/build_docx.py` — manuscript in the journal format (source not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

## Data

Operating-cost database: ANEEL, Nota Técnica 97/2022 and the annexes of Tomada de Subsídios 14/2022 (https://www.gov.br/aneel; public documents, open government data). Active fires by state: INPE BDQueimadas (https://terrabrasilis.dpi.inpe.br/queimadas; open data). The ANEEL spreadsheets (≈12 MB plus one file per concessionaire) are not redistributed; download them and place them under `dados/ts14_f1/` and `dados/extra/` as named in `scripts/painel.py`.

Raw input files are **not** redistributed in this repository; they are large and/or belong to third parties. Download them from the sources above and place them as described, or start from the derived tables in `resultados/`.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.
