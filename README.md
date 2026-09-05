# Software Effort Estimator

This repository contains a Streamlit research prototype for software-effort estimation. The COCOMO-81 path uses a training-derived FAHP-inspired proxy to rescale an RBF-kernel sklearn SVR, and KernelSHAP provides local model-output attributions in person-months.

The project is an adaptation inspired by Sehra et al. (2019), not a reproduction of their FAHP-weighted LSSVM. The current weights are supervised, data-derived proxies rather than an independent panel of expert judgements, and sklearn epsilon-SVR is not LSSVM.

## Research framing

The explainability extension asks whether the training-derived FAHP-inspired proxy agrees with the predictive reliance learned by the model. Merely adding SHAP is not claimed as novel: prior work has already applied model-agnostic explanations to SVR software-effort models and SHAP to simpler effort models. The pilot contribution here is the repeated, leakage-safe, out-of-fold comparison and its explicit unweighted comparator.

The primary comparison is:

> Training-derived FAHP-inspired proxy versus normalized mean absolute out-of-fold KernelSHAP reliance from an unweighted SVR.

The unweighted model is primary because the proxy weights were not injected into its kernel. Proxy-versus-weighted-model SHAP is reported only as a descriptive secondary comparison; its larger agreement is partly induced by construction.

## Explainability protocol and current evidence

The values below are stored under `metrics.json → models → cocomo81 → explainability`:

- 10 repeats of 5 outer folds, with 3-fold inner model selection.
- Preprocessing, proxy weights, and hyperparameters are fitted using outer-training projects only.
- Only outer-test projects are explained; each project is explained once per repeat.
- Each fold uses all outer-training projects as the KernelSHAP background; outcomes do not select background rows.
- The explainer evaluates the complete raw-input prediction pipeline with `nsamples=256`, explicit `l1_reg=0`, and no SHAP feature selection; all 17 features enter the weighted least-squares solution.
- Maximum stored additivity error: 0.000000 person-months.

| Comparison | Spearman ρ | Kendall τ-b | Jensen–Shannon similarity | Top-3 Jaccard / Kuncheva |
|---|---:|---:|---:|---:|
| Primary pooled: proxy vs unweighted OOF SHAP | 0.549020 | 0.352941 | 0.758709 | 0.200000 / 0.190476 (1/3 shared) |
| Primary median across repeats [min, max] | 0.471814 [0.404412, 0.651961] | 0.330882 [0.250000, 0.470588] | 0.755400 [0.734730, 0.771563] | 0.350000 [0.200000, 1.000000] / 0.392857 [0.190476, 1.000000] |
| Weighted pooled, descriptive/partly induced | 0.710784 | 0.500000 | 0.772099 | 0.500000 / 0.595238 (2/3 shared) |
| Weighted median across repeats [min, max] | 0.658088 [0.406863, 0.732843] | 0.470588 [0.308824, 0.558824] | 0.764367 [0.739879, 0.804626] | 0.500000 [0.500000, 0.500000] / 0.595238 [0.595238, 0.595238] |

A 2,000-resample project-level nonparametric percentile procedure gives a conditional 95% interval for primary Spearman ρ of **[0.401961, 0.607843]** (resampling median 0.514706). The resampling unit is a COCOMO project after averaging its repeated OOF explanations. It does not refit models, regenerate folds, retune hyperparameters, or rerun the explainer, so this is not a full uncertainty interval.

Primary unweighted stability is also reported across repeats:

- Joint repeat ranking stability: median Spearman ρ 0.921569 [0.811275, 0.968137], median Kendall τ-b 0.794118 [0.617647, 0.882353], and median top-3 Jaccard 0.500000 [0.200000, 1.000000] across 45 repeat pairs.
- Joint local-attribution stability: median cosine similarity 0.957340 [0.137241, 0.998259], median sign agreement 0.823529 [0.470588, 1.000000], and median top-3 Jaccard 0.500000 [0.000000, 1.000000] across 2,835 within-project repeat comparisons.

These are joint stability measurements: the split, fitted model, outer-training background, and KernelSHAP seed vary together. In a separate fixed-model numerical audit, 256 samples versus a 1,024-sample reference preserved the complete feature ranking (Spearman ρ = 1.000000), had Jensen–Shannon similarity 0.989210, identical top three, and median local cosine similarity 0.999547; seed-to-seed median global ρ was 0.997549. Every audited explanation retained all 17 features.

A 10,000-permutation feature-label benchmark found that pooled rank and magnitude agreement were above random relabelling (Spearman empirical p = 0.025797; Jensen–Shannon p = 0.003700), while top-3 overlap was not unusual under this narrow null (Jaccard p = 0.462754). This is descriptive evidence, not a population hypothesis test.

Repeated predictive accuracy is descriptive: the weighted model has lower MMRE in 9/10 runs. Median weighted versus unweighted values are MMRE 0.425209 versus 0.502545, MdMRE 0.307316 versus 0.361205, and MAE 208.131571 versus 245.059008 person-months. Median paired differences (unweighted minus weighted) are +0.080563 MMRE, +0.037036 MdMRE, and +19.813137 person-months MAE. Repeated partitions overlap and are not independent inferential units, so no uncorrected significance test is reported.

The five highest features by unweighted OOF SHAP reliance are:

| Feature | Proxy weight (rank) | Unweighted OOF SHAP reliance (rank) | Weighted OOF SHAP reliance (rank) |
|---|---:|---:|---:|
| LOC (KLOC) | 0.226659 (#1) | 0.329450 (#1) | 0.373351 (#1) |
| RELY | 0.044488 (#7) | 0.072291 (#2) | 0.059705 (#5) |
| TIME | 0.041998 (#8) | 0.069630 (#3) | 0.068650 (#3) |
| Dev Mode | 0.114108 (#3) | 0.069218 (#4) | 0.091528 (#2) |
| PCAP | 0.054818 (#6) | 0.067262 (#5) | 0.067857 (#4) |

These are model-reliance and stability measurements, not causal effects. SHAP depends on the chosen background, correlated COCOMO drivers can redistribute attribution, and low stability minima mean a single explanation should not be presented as universally stable. The interactive app uses a separate, outcome-conditioned 50-project background for responsiveness; its base value is the expected model output over those rows, not a representative population mean.

The weighting proxy is not independent expert FAHP. On the full data, Chang extent analysis leaves 25/63 effort project weights, 23/63 LOC project weights, and 30/63 combined project weights nonzero at the recorded threshold; the effort pairwise consistency ratio is 0.105267 and the LOC ratio is 0.054688. Proxy-weight sensitivity and a frozen expert-elicited matrix are required for a confirmatory extension with Professor Sehra.

## Research references

- Sehra, S. K., Brar, Y. S., Kaur, N., & Sehra, S. S. (2019). “Software effort estimation using FAHP and weighted kernel LSSVM machine.” *Soft Computing, 23*(21), 10881–10900. [https://doi.org/10.1007/s00500-018-3639-2](https://doi.org/10.1007/s00500-018-3639-2)
- Lundberg, S. M., & Lee, S.-I. (2017). “A unified approach to interpreting model predictions.” *Advances in Neural Information Processing Systems, 30*.
- Najm, A., Zakrani, A., & Marzak, A. (2023). “Efficient Shapely Explanation of Support Vector Regression for Agile and Non-agile Software Effort Estimation.” In *Intelligent Sustainable Systems*, 711–729. [https://doi.org/10.1007/978-981-19-7663-6_67](https://doi.org/10.1007/978-981-19-7663-6_67)
- Jayadi, P., Prasetya, D. D., & Widiyaningtyas, T. (2025). “Explanation of a Simple Machine Learning Model for Software Effort Estimation using SHAP (SHapley Additive exPlanations).” *2025 9th International Conference on Electrical, Electronics and Information Engineering (ICEEIE)*, 1–5. [https://doi.org/10.1109/ICEEIE66203.2025.11254827](https://doi.org/10.1109/ICEEIE66203.2025.11254827)

## Quick local run

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (recommended)

1. Push this repository to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-git-repo-url>
git push -u origin main
```

2. Go to https://share.streamlit.io, sign in with GitHub, and create a new app by selecting this repository and the `app.py` file. Streamlit Cloud will install dependencies from `requirements.txt` and deploy the app.

## Deploy to Heroku (alternative)

1. Install the Heroku CLI and login:

```bash
heroku login
```

2. Create a Heroku app and push:

```bash
heroku create your-app-name
git push heroku main
```

3. Ensure the `Procfile` is present (included) so Heroku runs Streamlit correctly.

## Notes

- Ensure `model.pkl` is committed (or use Git LFS if large).
- If deployment fails due to binary packages (e.g., `shap`), consider building a smaller runtime or using Docker.
