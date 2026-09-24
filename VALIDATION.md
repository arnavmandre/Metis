# Verification record

- Five integrity unit tests passed.
- Full synthetic CLI pipeline completed, including all three model families, validation selection, held-out backtests, robustness experiments and 16 figures.
- Notebook validated against nbformat and all code cells executed sequentially in one Python process from its embedded source bundle.
- A separate Jupyter kernel could not start because this environment prohibits its socket operation. Hosted Kaggle execution has not been directly verified.
- Historical Yahoo download attempted; SPY failed with rate limiting and a network timeout. No historical results are included or inferred from synthetic data.
- Synthetic output is reproducible with seed 42. The regime portfolio reduced drawdown but had a worse Sharpe ratio than both included benchmarks in this demonstration.

No real-world predictive or investment performance is established by these checks.
