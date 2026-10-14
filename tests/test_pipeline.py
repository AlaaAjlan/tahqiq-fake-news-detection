import os
import sys

import joblib
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_preprocessing import clean_text  # noqa: E402


def test_clean_text_removes_urls_and_agency_leakage():
    text = "WASHINGTON (Reuters) - Read more at https://example.com NOW!!"
    assert clean_text(text) == "washington read more at now"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


@pytest.mark.parametrize("version", ["baseline", "adapted"])
def test_saved_models_load_and_predict(version):
    tfidf = joblib.load(os.path.join(ROOT, f"{version}_tfidf.pkl"))
    model = joblib.load(os.path.join(ROOT, f"{version}_model.pkl"))
    X = tfidf.transform([clean_text("The president signed the new bill into law on Tuesday.")])
    proba = model.predict_proba(X)[0]
    assert len(proba) == 2
    assert abs(sum(proba) - 1.0) < 1e-6
    assert model.predict(X)[0] in (0, 1)
