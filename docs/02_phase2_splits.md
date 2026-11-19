# Phase 2: leakage-safe preprocessing and splits

Run: `python -m ml.preprocessing.build` (about 20 s on CPU). Artefacts go to `data/processed/`
(not committed; rebuild from the raw CSVs, and `manifest.json` records their sha256).

## Splits (labelled nodes only; computed by the build, copied to `reports/split_summary.csv`)

| Split | Train steps | Val steps | Test steps | Train labelled (illicit %) | Val labelled (illicit %) | Test labelled (illicit %) |
|---|---|---|---|---|---|---|
| paper | 1–29 | 30–34 | 35–49 | 26,381 (10.9%) | 3,513 (16.8%) | 16,670 (6.5%) |
| paper_refit | 1–34 | — | 35–49 | 29,894 (11.6%) | — | 16,670 (6.5%) |
| fc_24 | 1–19 | 20–24 | 25–29 | 17,989 (8.4%) | 5,617 (12.6%) | 2,775 (23.5%) |
| fc_29 | 1–24 | 25–29 | 30–34 | 23,606 (9.4%) | 2,775 (23.5%) | 3,513 (16.8%) |
| fc_34 | 1–29 | 30–34 | 35–39 | 26,381 (10.9%) | 3,513 (16.8%) | 5,486 (8.1%) |
| fc_39 | 1–34 | 35–39 | 40–44 | 29,894 (11.6%) | 5,486 (8.1%) | 7,458 (6.9%) |
| fc_44 | 1–39 | 40–44 | 45–49 | 35,380 (11.0%) | 7,458 (6.9%) | 3,726 (3.2%) |

## Checks run on the real data

- Every split: training-period features standardise to mean 0 (max |mean| ≤ 4e-4).
  Test-period features are *not* re-centred (max |mean| 2.4–3.5 standard deviations),
  which is expected: the test period is scaled with training statistics only.
- Every split: the training graph contains 0 test-period nodes.

## Notes for later phases

- **Illicit rates swing a lot between windows** (3.2% to 23.5% across test windows).
  Precision and F1 aren't comparable across windows without accounting for this;
  PR-AUC should be read against each window's base rate (the "no-skill" line).
- **fc_44's test window has only 121 illicit nodes.** Bootstrap CIs will be wide;
  report them.
- `paper` and `fc_34` share the same training and validation steps, so one tuning
  run serves both.
