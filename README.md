# Software Effort Estimator

This repository contains a Streamlit research prototype for software-effort estimation. The COCOMO-81 path uses a training-derived FAHP-inspired proxy to rescale an RBF-kernel sklearn SVR, and KernelSHAP provides local model-output attributions in person-months.

The project is an adaptation inspired by Sehra et al. (2019), not a reproduction of their FAHP-weighted LSSVM. The current weights are supervised, data-derived proxies rather than an independent panel of expert judgements, and sklearn epsilon-SVR is not LSSVM.

## Research framing

The explainability extension asks whether the training-derived FAHP-inspired proxy agrees with the predictive reliance learned by the model. Merely adding SHAP is not claimed as novel: prior work has already applied model-agnostic explanations to SVR software-effort models and SHAP to simpler effort models. The pilot contribution here is the repeated, leakage-safe, out-of-fold comparison and its explicit unweighted comparator.

The primary comparison is:

> Training-derived FAHP-inspired proxy versus normalized mean absolute out-of-fold KernelSHAP reliance from an unweighted SVR.

The unweighted model is primary because the proxy weights were not injected into its kernel. Proxy-versus-weighted-model SHAP is reported only as a descriptive secondary comparison; its larger agreement is partly induced by construction.

## Comparison with standard estimators

The first question for any new effort model is whether it beats the established ones. `baselines.py` scores the SVR against intermediate COCOMO-81, COCOMO-81 recalibrated on each training fold, log-linear regression (ridge, penalty chosen by inner CV), and the training median. Every model uses the same 10 outer folds as the reported results, and anything fitted is fitted on outer-training projects only. SA is Standardised Accuracy against random guessing (Shepperd & MacDonell, 2012), and higher is better. MAE and MdAE are in person-months.

**COCOMO-81, 10-fold cross-validation (63 projects)**

| Model | MAE | MdAE | MMRE | MdMRE | Pred(25) | SA |
|---|---:|---:|---:|---:|---:|---:|
| SVR, FAHP-proxy weighted (this project) | **149.9** | 24.4 | 0.391 | 0.308 | 0.46 | **0.87** |
| SVR, unweighted | 213.1 | 28.9 | 0.431 | 0.324 | 0.38 | 0.81 |
| Intermediate COCOMO-81, published coefficients\* | 184.5 | **20.7** | **0.323** | **0.222** | **0.56** | 0.84 |
| COCOMO-81 recalibrated per fold | 190.1 | 33.8 | 0.366 | 0.243 | 0.51 | 0.83 |
| Log-linear regression (ridge, nested) | 243.2 | 34.3 | 0.413 | 0.299 | 0.44 | 0.78 |
| Training median effort | 648.1 | 86.0 | 2.529 | 0.887 | 0.08 | 0.42 |

\*Boehm derived these coefficients from these same 63 projects, so this row is effectively scored on its own calibration data. The recalibrated row is the fair out-of-sample version.

**NASA93, external test (93 projects; every model fitted on all of COCOMO-81)**

| Model | MAE | MdAE | MMRE | MdMRE | Pred(25) | SA |
|---|---:|---:|---:|---:|---:|---:|
| SVR, FAHP-proxy weighted (this project) | **274.5** | **43.6** | **0.518** | 0.296 | 0.46 | **0.68** |
| Intermediate COCOMO-81, published coefficients | 394.6 | 49.0 | 0.585 | **0.253** | **0.48** | 0.54 |
| COCOMO-81 recalibrated | 371.3 | 48.2 | 0.577 | 0.288 | 0.40 | 0.56 |
| Log-linear regression (ridge) | 399.9 | 55.6 | 0.567 | 0.295 | 0.43 | 0.53 |
| Training median effort | 558.6 | 154.0 | 1.195 | 0.786 | 0.10 | 0.34 |

Paired comparisons use per-project absolute errors: a Wilcoxon signed-rank test, and a 95% bootstrap interval for the MAE difference (projects resampled, models not refitted). Negative means the weighted SVR is better.

| Weighted SVR vs | COCOMO-81 CV: MAE difference [95% CI], p | NASA93: MAE difference [95% CI], p |
|---|---|---|
| Published COCOMO-81 | −34.6 [−154.7, +77.5], p = 0.81 | −120.1 [−345.5, +8.8], p = 0.56 |
| Recalibrated COCOMO-81 | −40.2 [−183.9, +59.2], p = 0.52 | **−96.8 [−254.4, −8.6], p = 0.012** |
| Log-linear regression | −93.4 [−257.3, +1.0], p = 0.19 | −125.3 [−339.0, −9.4], p = 0.10 |
| Unweighted SVR | −63.2 [−192.2, +25.9], p = 0.15 | – |

**What this shows:**
- On COCOMO-81, the weighted SVR has the lowest absolute error, but it is not reliably better than the standard models. The published COCOMO equation has better relative-error scores there, helped by having been fitted to these same projects.
- On the external NASA93 projects, the SVR transfers best. Its MAE is about 100 person-months lower than every standard model, and it is significantly better than recalibrated COCOMO.
- The headline claim is therefore better generalisation to unseen projects, not better accuracy on COCOMO-81.

MMRE is reported for comparability with earlier work, but it favours models that underestimate (Foss et al., 2003), so the absolute-error measures and SA should be read first.

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
- Boehm, B. W. (1981). *Software Engineering Economics*. Prentice Hall.
- Foss, T., Stensrud, E., Kitchenham, B., & Myrtveit, I. (2003). “A simulation study of the model evaluation criterion MMRE.” *IEEE Transactions on Software Engineering, 29*(11), 985–995.
- Shepperd, M., & MacDonell, S. (2012). “Evaluating prediction systems in software project estimation.” *Information and Software Technology, 54*(8), 820–827.
- Langdon, W. B., Dolado, J., Sarro, F., & Harman, M. (2016). “Exact mean absolute error of baseline predictor, MARP0.” *Information and Software Technology, 73*, 16–18.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py          # the estimator app
python train_model.py         # retrain the COCOMO-81 models; rewrites metrics.json and model.pkl
python baselines.py           # comparison with standard estimators; writes baselines.json (~30 s)
pip install pytest
python -m pytest -q           # 31 tests
```

## Project layout

```text
app.py, pages/            Streamlit app: estimator, GitHub-data model, intake of your own data
train_model.py            nested CV, weighted vs unweighted SVR, NASA93 external test, SHAP extension
fahp.py                   Chang (1996) extent-analysis FAHP and the training-derived weighting proxy
validation.py             leakage-safe model selection, metrics, prediction intervals
baselines.py              standard comparators on the same folds; paired tests
explainability.py         repeated out-of-fold KernelSHAP analysis
collect_github.py, train_github_model.py   the 91-repository GitHub proof of concept
metrics.json, baselines.json               every reported number
tests/                    unit tests
```
