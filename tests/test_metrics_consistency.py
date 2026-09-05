import json
import pickle
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_model_artifacts_match_metrics_json():
    metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))["models"]
    with (ROOT / "model.pkl").open("rb") as handle:
        cocomo_artifact = pickle.load(handle)
    with (ROOT / "github_model.pkl").open("rb") as handle:
        github_artifact = pickle.load(handle)

    assert cocomo_artifact["loo_mmre"] == pytest.approx(metrics["cocomo81"]["nested_cv"]["mmre"], abs=5e-5)
    assert cocomo_artifact["pred25"] == pytest.approx(metrics["cocomo81"]["nested_cv"]["pred25"], abs=5e-5)
    assert cocomo_artifact["explainability"] == metrics["cocomo81"]["explainability"]
    assert github_artifact["loo_mmre"] == pytest.approx(metrics["github"]["nested_cv"]["mmre"], abs=5e-5)
    assert github_artifact["pred25"] == pytest.approx(metrics["github"]["nested_cv"]["pred25"], abs=5e-5)


def test_active_model_pages_use_metrics_json_for_scores():
    estimator = (ROOT / "pages" / "1_Estimator.py").read_text(encoding="utf-8")
    github = (ROOT / "pages" / "2_GitHub_Model.py").read_text(encoding="utf-8")
    for source in (estimator, github):
        assert "from app_metrics import model_metrics" in source
        assert "LOO-MMRE" not in source
        assert "Temporal MMRE" not in source
        assert "use_container_width" not in source
    for misleading_label in (
        "FAHP Prior",
        "SHAP Posterior",
        "Alignment Score",
        "Top recommendation",
        "GenAI saves",
        "AI Saving",
        "Sehra et al. (2018)",
    ):
        assert misleading_label not in estimator
    assert "Same methodology as the COCOMO-81 model" not in github
    assert "FAHP Feature Weights" not in github


def test_streamlit_version_is_exactly_pinned():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "streamlit==1.63.0" in requirements
    assert "shap==0.52.0" in requirements


def test_research_documentation_uses_correct_foundational_citation():
    generator = (ROOT / "generate_pdf.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for source in (generator, readme):
        assert "Software effort estimation using FAHP and weighted kernel LSSVM machine" in source
        assert "10.1007/s00500-018-3639-2" in source
        assert "10.1007/978-981-19-7663-6_67" in source
        assert "10.1109/ICEEIE66203.2025.11254827" in source
        assert "Sehra et al. (2018)" not in source
