# Frozen retrospective holdout protocol

This protocol was written before the first historical test evaluation. The model, source hashes, data hashes and settings are saved by `blind.py fit` in `outputs/blind/freeze.json`. A Git commit records that freeze before `blind.py evaluate` is invoked.

1. Download the daily factor archive from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html). Recover market total returns as `(Mkt-RF + RF) / 100`; use `RF / 100` for the defensive T-bill research index. Never use a yield as a price return.
2. Use 1999 for feature warm-up; fit on 2000–2014; choose 2/3/4 HMM states on 2015–2019 validation likelihood. For each state count, choose the best of seeds 7/42/99 by training likelihood. All scaling, state-risk ranking and forecast calibration use training data only.
3. Freeze all settings before revealing 2020–2025. The 2026 observations in the source are intentionally excluded from this experiment and remain available for a future, separately designed check.
4. Define the named binary target as future 20-session realized volatility above the training 75th percentile. Remove the last 20 training observations from supervised calibration because their outcomes cross the split. Use a fixed 0.5 classification threshold.
5. Compare the HMM-probability logistic forecasting head against a logistic model using trailing 20-day volatility, and the constant training event rate. Quote accuracy, balanced accuracy, precision, recall, F1, ROC AUC, Brier and log loss. Quote no real-regime accuracy because there are no definitive labels for financial regimes.
6. Calculate 95% paired moving-block bootstrap intervals: 1,000 resamples, 60-session blocks, seed 1907. Outcomes overlap, so ordinary IID accuracy intervals would be misleading. The intervals condition on the chosen model and cannot prove future stability.
7. Test portfolios with 5 bps commission + 5 bps slippage per traded leg, five-session rebalance cadence, a 5-point equity trade band and next-close execution. Compare equity buy-and-hold, 60/40, validation-exposure-matched, simple 10%-vol-target and unguarded/guarded regime strategies. Delay/slippage stresses are prespecified diagnostics, not strategy selection.
8. Separately train a known-state synthetic recovery experiment on seed 11 and evaluate untouched seeds 97, 123 and 2026. Learn state-to-truth mapping on training data only; do not remap against test labels.
9. Report failures, calibration, distribution drift, yearly performance and differences from baselines. Do not tune after seeing the holdout. Changes prompted by test results would require a new untouched dataset.

## Scope of the word “blind”

Blind to model fitting and selection, with a code/data/model freeze before evaluation. Not externally administered, not prospective, and not free of a researcher's prior knowledge of the COVID and rate-hike periods. Current French research data incorporate revisions and are published after observation dates. Simulated portfolio returns are research proxies, not evidence that these prices and features could have been traded at the stated times.

## Commands

```bash
python blind.py prepare             # download + seal chronological partitions
python blind.py fit                 # fit development data only; save model + freeze
python blind.py evaluate            # reveal holdout once; refuse silent overwrite
python paper_signal.py --csv your_complete_price_history.csv
```

Keep the exact archive: its checksum is recorded, and future provider revisions can change results. Do not unseal/tune on 2026 as part of this test. `paper_signal.py` is read-only and suppresses its paper target for stale or out-of-distribution data. A new live feed/instrument still needs its own validation.
