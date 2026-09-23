import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.models import artifacts, predict
from src.models.estimators.xgboost import XGBoostEstimator

FAST_XGB = dict(
    objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
    enable_categorical=True, n_estimators=25, max_depth=3,
    random_state=42, n_jobs=1, early_stopping_rounds=10,
)

TERMINAL_MAP = pd.DataFrame(
    {
        "terminal_code": np.array([100, 200, 300], dtype="int32"),
        "terminal_id": np.array([0, 1, 2], dtype="int64"),
    }
)


class _StubEstimator:
    """Stands in for a fitted estimator so the tests stay deterministic."""

    def get_feature_importances(self, feature_columns):
        return np.array([2.0, 1.0]), "split"

    def predict(self, X):
        # deliberately exceeds [0, 1] to prove the clipping is applied
        return X["f1"].to_numpy() * 5.0


def _bundle():
    return {
        "model": _StubEstimator(),
        "model_name": "lightgbm",
        "horizon": "h1h",
        "target": "y_h1h",
        "feature_columns": ["f1", "terminal_id"],
        "terminal_map": TERMINAL_MAP,
        "categorical_features": ("terminal_id",),
    }


def test_feature_importance_is_sorted_and_carries_its_type(tmp_path):
    path = artifacts.save_feature_importance(
        tmp_path, _StubEstimator(), "lightgbm", "h1h", ["f1", "terminal_id"]
    )

    table = pl.read_csv(path)

    assert path.name == "lightgbm_h1h_feature_importance.csv"
    assert table.columns == ["feature", "importance", "importance_type"]
    assert table["feature"].to_list() == ["f1", "terminal_id"]
    assert table["importance"].to_list() == [2.0, 1.0]
    assert table["importance_type"].unique().to_list() == ["split"]


def test_feature_importance_rejects_a_length_mismatch(tmp_path):
    """A misaligned mapping would silently label the wrong features."""
    with pytest.raises(ValueError, match="importance values for"):
        artifacts.save_feature_importance(
            tmp_path, _StubEstimator(), "xgboost", "h1h", ["a", "b", "c"]
        )


def test_load_bundle_rejects_an_incomplete_artifact(tmp_path):
    path = tmp_path / "partial.joblib"
    joblib.dump({"model": object(), "model_name": "lightgbm"}, path)

    with pytest.raises(ValueError, match="missing bundle keys"):
        predict.load_bundle(path)


def test_load_bundle_round_trips_a_complete_artifact(tmp_path):
    path = artifacts.save_model_bundle(
        model_dir=tmp_path,
        model=_StubEstimator(),
        model_name="lightgbm",
        horizon_name="h1h",
        target="y_h1h",
        feature_columns=["f1", "terminal_id"],
        terminal_map=TERMINAL_MAP,
    )

    bundle = predict.load_bundle(path)

    assert path.name == "lightgbm_h1h.joblib"
    assert bundle["horizon"] == "h1h"
    assert bundle["feature_columns"] == ["f1", "terminal_id"]


def test_predict_frame_preserves_row_order_and_clips():
    df = pl.DataFrame(
        {
            "terminal_code": np.array([200, 100, 200], dtype="int32"),
            "f1": [0.1, 0.2, 0.4],
        }
    )

    out = predict.predict_frame(_bundle(), df)

    assert out.columns == ["terminal_code", "f1", "pred_h1h"]
    assert out["terminal_code"].to_list() == [200, 100, 200]
    # 0.1*5, 0.2*5, 0.4*5 -> 0.5, 1.0, 2.0 clipped to 1.0
    assert out["pred_h1h"].to_list() == pytest.approx([0.5, 1.0, 1.0])


def test_predict_rejects_a_terminal_absent_from_training():
    df = pl.DataFrame(
        {
            "terminal_code": np.array([100, 999], dtype="int32"),
            "f1": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="not seen during training: \\[999\\]"):
        predict.predict_occupancy(_bundle(), df)


def test_predict_rejects_a_frame_without_the_required_features():
    df = pl.DataFrame({"terminal_code": np.array([100], dtype="int32")})

    with pytest.raises(ValueError, match="missing required features"):
        predict.predict_occupancy(_bundle(), df)


def test_saved_bundle_can_predict_after_a_reload(tmp_path):
    """Regression: the bundle must carry the estimator, not the raw XGBRegressor.

    A raw model holds no category mapping, so terminal_id reaches predict() as
    int64 while the booster was trained on a categorical column.
    """
    r = np.random.default_rng(0)
    X_tr = pd.DataFrame(
        {"f0": r.random(300), "terminal_id": r.choice([0, 1, 2], 300).astype("int64")}
    )
    y_tr = pd.Series(r.random(300))
    X_va = pd.DataFrame(
        {"f0": r.random(100), "terminal_id": r.choice([0, 1], 100).astype("int64")}
    )
    y_va = pd.Series(r.random(100))

    est = XGBoostEstimator(name="xgboost", params=dict(FAST_XGB))
    est.fit(X_tr, y_tr, X_va, y_va, ["terminal_id"])

    path = artifacts.save_model_bundle(
        model_dir=tmp_path,
        model=est,
        model_name="xgboost",
        horizon_name="h1h",
        target="y_h1h",
        feature_columns=["f0", "terminal_id"],
        terminal_map=TERMINAL_MAP,
    )

    bundle = predict.load_bundle(path)
    df = pl.DataFrame(
        {
            "terminal_code": np.array([300, 100, 200], dtype="int32"),
            "f0": [0.3, 0.7, 0.5],
        }
    )

    out = predict.predict_frame(bundle, df)

    assert out["pred_h1h"].is_not_null().all()
    assert out["pred_h1h"].min() >= 0.0
    assert out["pred_h1h"].max() <= 1.0
