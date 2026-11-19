# Project 1 status
Completed 2026-11-20: full 4.85M daily data, 14 SQL models, original 5 rolling folds and full final grid, 1,000-store bootstrap, calibration, slices, SHAP, validation permutation importance and last-day unlabeled inference.

New work completed: business marts and integrity checks, matched-budget allocation at K=1/3/5/10, paired 1,000-store bootstrap for K=5, offline Chinese dashboard, report, resume/interview guide and reproducible code archive. Original tests 43 passed; new tests 11 passed.

UI verified: city/date filters match CSV totals, invalid date handling, store selection, capacity switching, desktop and 390px layout; no browser console errors. Shared Plotly axis mutation found and corrected during QA.

Measured: model P@10%=0.6513714286 vs today's-rule 0.5756285714; R@10%=0.188573. Primary matched budget 4,486/day: global coverage 0.964365, floor policy 1.0; P=0.657984 vs 0.658143, paired precision difference CI includes zero. No causal business improvement or real deployment claimed.
