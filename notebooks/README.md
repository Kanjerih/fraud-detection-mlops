# Notebooks

Drop the original exploratory notebook here as `01-eda-and-baseline.ipynb`
(the one with the EDA, correlation heatmap, initial Random Forest baseline,
stratified-split fix, cross-validation, and threshold/savings analysis).

Notebooks in this project are for **exploration only**. Nothing in
`notebooks/` is imported by `src/` or run in CI — once an approach is
validated here, its logic is ported into `src/` as tested, versioned code
(see `src/models/train.py` and `src/models/evaluate.py`, which are the
production versions of the analysis originally done in this notebook).
