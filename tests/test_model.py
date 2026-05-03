import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must come before other mpl imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import src.Model as model_module
from src.Model import Model


@pytest.fixture
def sample_data():
    np.random.seed(0)
    n_train, n_test, n_feat = 120, 30, 10
    cols = [f"f{i}" for i in range(n_feat)]
    X_train = pd.DataFrame(np.random.randn(n_train, n_feat), columns=cols)
    y_train = pd.Series(np.random.randint(0, 2, n_train))
    X_test  = pd.DataFrame(np.random.randn(n_test, n_feat),  columns=cols)
    y_test  = pd.Series(np.random.randint(0, 2, n_test))
    return X_train, y_train, X_test, y_test


@pytest.fixture
def biz_stats():
    return pd.DataFrame({
        "avg_loan_profit": [300.0],
        "avg_loan_loss":   [850.0],
        "avg_loan_amount": [1000.0],
    })


@pytest.fixture
def model(sample_data, biz_stats, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = sample_data
    return Model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        business_statistics=biz_stats,
        output_dir=str(tmp_path / "out"),
        random_search_iter=1,
    )


@pytest.fixture
def fitted_lr(model):
    return _make_mock_estimator(
        "LogisticRegression",
        {"C": 1.0, "solver": "lbfgs", "max_iter": 200, "random_state": 42},
    )


def _make_result(name="Model A", stage="test", acc=0.80, prec=0.78,
                 rec=0.75, f2=0.76, roc_auc=0.82, y_prob=None,
                 cm=None, report="", model_obj=None):
    if y_prob is None:
        y_prob = np.full(20, 0.6)
    if cm is None:
        cm = np.array([[14, 2], [3, 1]])
    return dict(name=name, stage=stage, acc=acc, precision=prec,
                recall=rec, f2=f2, roc_auc=roc_auc, y_prob=y_prob,
                cm=cm, report=report, model=model_obj)


def _mock_scores(n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.array([])
    if n_samples == 1:
        return np.array([0.0])
    return np.linspace(-1.0, 1.0, n_samples)


def _make_mock_estimator(name: str, params: dict | None = None):
    mock_estimator = MagicMock(name=name)
    if params is None:
        params = {"mock_name": name}
    mock_estimator.get_params.return_value = params
    mock_estimator._cv_best_score = 0.91
    mock_estimator._cv_best_params = {"mock_name": name}

    def _predict(X):
        scores = _mock_scores(len(X))
        return (scores >= 0).astype(int)

    def _predict_proba(X):
        scores = _mock_scores(len(X))
        probabilities = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probabilities, probabilities])

    def _decision_function(X):
        return _mock_scores(len(X))

    mock_estimator.predict.side_effect = _predict
    mock_estimator.predict_proba.side_effect = _predict_proba
    mock_estimator.decision_function.side_effect = _decision_function
    return mock_estimator


class _FakeSelectKBest:
    def __init__(self, score_func=None, k=10):
        self.score_func = score_func
        self.k = k
        self._selected_mask = None

    def fit_transform(self, X, y):
        self._set_mask(X)
        return np.asarray(X.iloc[:, self._selected_indices])

    def transform(self, X):
        return np.asarray(X.iloc[:, self._selected_indices])

    def get_support(self):
        return self._selected_mask

    def _set_mask(self, X):
        n_features = X.shape[1]
        k = min(self.k, n_features)
        self._selected_indices = list(range(k))
        self._selected_mask = np.zeros(n_features, dtype=bool)
        self._selected_mask[self._selected_indices] = True


class _FakeRandomizedSearchCV:
    def __init__(self, estimator, param_distributions, n_iter, scoring, cv,
                 random_state, n_jobs, verbose):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        estimator_name = self.estimator.__class__.__name__
        params = {}
        if hasattr(self.estimator, "get_params"):
            try:
                params = self.estimator.get_params()
            except Exception:
                params = {"mock_name": estimator_name}
        self.best_estimator_ = _make_mock_estimator(estimator_name, params)
        self.best_score_ = 0.91
        self.best_params_ = {"mock_name": estimator_name}
        return self


@pytest.fixture(autouse=True)
def _mock_external_modeling_dependencies(monkeypatch):
    monkeypatch.setattr(model_module, "SelectKBest", _FakeSelectKBest)
    monkeypatch.setattr(model_module, "RandomizedSearchCV", _FakeRandomizedSearchCV)


class TestInit:
    def test_data_shapes_stored(self, model, sample_data):
        X_train, y_train, X_test, y_test = sample_data
        assert model.X_train.shape == X_train.shape
        assert model.y_train.shape == y_train.shape
        assert model.X_test.shape  == X_test.shape
        assert model.y_test.shape  == y_test.shape

    def test_business_statistics_stored(self, model, biz_stats):
        assert model.business_statistics.equals(biz_stats)

    def test_output_dir_created(self, model):
        assert model.output_dir.exists()
        assert model.output_dir.is_dir()

    def test_plots_dir_is_child_of_output(self, model):
        assert model.plots_dir.exists()
        assert model.plots_dir.parent == model.output_dir

    def test_random_search_iter_stored(self, model):
        assert model.random_search_iter == 1

    def test_mlflow_experiment_name(self, model):
        assert model.mlflow_experiment_name == "Loan_Status_Classification"

    def test_mlflow_tracking_uri_is_sqlite(self):
        uri = mlflow.get_tracking_uri()
        assert uri.startswith("sqlite:///")

    def test_logger_is_logging_logger(self, model):
        assert isinstance(model.logger, logging.Logger)

    def test_logger_has_handlers(self, model):
        root_handlers = logging.getLogger().handlers
        named_handlers = model.logger.handlers
        total = len(root_handlers) + len(named_handlers)
        assert total > 0

    def test_log_file_created(self, model, tmp_path):
        log_file = tmp_path / "logs" / "ModelDevelopment.log"
        assert log_file.exists()

    def test_output_dir_path_object(self, model):
        assert isinstance(model.output_dir, Path)
        assert isinstance(model.plots_dir, Path)

class TestExtractHyperparameters:
    def test_returns_dict(self, model, fitted_lr):
        result = model._extract_hyperparameters("LR", fitted_lr)
        assert isinstance(result, dict)

    def test_contains_params(self, model, fitted_lr):
        result = model._extract_hyperparameters("LR", fitted_lr)
        assert len(result) > 0

    def test_all_values_are_strings(self, model, fitted_lr):
        result = model._extract_hyperparameters("LR", fitted_lr)
        for v in result.values():
            assert isinstance(v, str)

    def test_no_estimator_double_underscore_keys(self, model, fitted_lr):
        result = model._extract_hyperparameters("LR", fitted_lr)
        for k in result:
            assert not k.startswith("estimator__")

    def test_cv_best_params_prefixed_with_best(self, model):
        m = MagicMock()
        m.get_params.return_value = {"C": 1.0}
        m._cv_best_params = {"C": 0.5, "solver": "lbfgs"}
        result = model._extract_hyperparameters("Mock", m)
        assert "best_C" in result
        assert "best_solver" in result

    def test_model_without_get_params(self, model):
        m = MagicMock(spec=[])
        result = model._extract_hyperparameters("Empty", m)
        assert isinstance(result, dict)

    def test_adaboost_nested_params_excluded(self, model):
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=50, random_state=42
        )
        result = model._extract_hyperparameters("AdaBoost", ada)
        for k in result:
            assert not k.startswith("estimator__")

class TestComputeBusinessMetrics:
    def test_returns_all_six_keys(self, model):
        cm = np.array([[80, 10], [5, 5]])
        keys = model._compute_business_metrics(cm).keys()
        for k in ("false_positive_rate", "true_positive_rate",
                  "avg_missed_profit", "avg_gained_profit",
                  "avg_lost_loans_amount", "avg_saved_loans_amnts"):
            assert k in keys

    def test_fpr_formula(self, model):
        cm = np.array([[100, 20], [10, 20]])   # tn=100, fp=20
        metrics = model._compute_business_metrics(cm)
        assert pytest.approx(metrics["false_positive_rate"], rel=1e-4) == 20 / 120

    def test_tpr_formula(self, model):
        cm = np.array([[100, 20], [10, 30]])   # fn=10, tp=30
        metrics = model._compute_business_metrics(cm)
        assert pytest.approx(metrics["true_positive_rate"], rel=1e-4) == 30 / 40

    def test_fpr_zero_when_no_negatives_predicted(self, model):
        cm = np.array([[100, 0], [0, 20]])
        assert model._compute_business_metrics(cm)["false_positive_rate"] == 0

    def test_tpr_one_when_all_positives_caught(self, model):
        cm = np.array([[100, 0], [0, 20]])
        assert model._compute_business_metrics(cm)["true_positive_rate"] == 1.0

    def test_all_fp_zero_division(self, model):
        # tn=0, fp=0 → fpr should be 0 (guarded division)
        cm = np.array([[0, 0], [5, 15]])
        metrics = model._compute_business_metrics(cm)
        assert metrics["false_positive_rate"] == 0

    def test_all_fn_zero_division(self, model):
        # fn=0, tp=0 → tpr should be 0
        cm = np.array([[50, 10], [0, 0]])
        metrics = model._compute_business_metrics(cm)
        assert metrics["true_positive_rate"] == 0

    def test_avg_missed_profit(self, model):
        cm = np.array([[80, 20], [5, 5]])      # fp=20
        metrics = model._compute_business_metrics(cm)
        assert pytest.approx(metrics["avg_missed_profit"]) == 20 * 300.0

    def test_avg_gained_profit(self, model):
        cm = np.array([[80, 20], [5, 5]])      # tn=80
        metrics = model._compute_business_metrics(cm)
        assert pytest.approx(metrics["avg_gained_profit"]) == 80 * 300.0

    def test_avg_lost_loans_amount(self, model):
        cm = np.array([[80, 20], [7, 5]])      # fn=7
        metrics = model._compute_business_metrics(cm)
        assert pytest.approx(metrics["avg_lost_loans_amount"]) == 7 * 1000.0

    def test_avg_saved_loans_amnts(self, model):
        cm = np.array([[80, 20], [7, 9]])      # tp=9
        metrics = model._compute_business_metrics(cm)
        assert pytest.approx(metrics["avg_saved_loans_amnts"]) == 9 * 1000.0

    def test_perfect_predictions(self, model):
        cm = np.array([[100, 0], [0, 50]])
        m = model._compute_business_metrics(cm)
        assert m["avg_missed_profit"]     == 0
        assert m["avg_lost_loans_amount"] == 0


# ──────────────────────────────────────────────
# 4. _save_matplotlib_plot
# ──────────────────────────────────────────────

class TestSaveMatplotlibPlot:
    def test_file_created_in_plots_dir(self, model):
        plt.figure()
        plt.plot([1, 2, 3])
        model._save_matplotlib_plot("test.png")
        assert (model.plots_dir / "test.png").exists()

    def test_figure_closed_after_save(self, model):
        plt.figure()
        plt.plot([0, 1])
        before = plt.get_fignums()
        model._save_matplotlib_plot("close_test.png")
        after = plt.get_fignums()
        assert len(after) < len(before)

    def test_different_filenames_create_different_files(self, model):
        for name in ("a.png", "b.png", "c.png"):
            plt.figure()
            plt.plot([1])
            model._save_matplotlib_plot(name)
        for name in ("a.png", "b.png", "c.png"):
            assert (model.plots_dir / name).exists()


# ──────────────────────────────────────────────
# 5. _select_features
# ──────────────────────────────────────────────

class TestSelectFeatures:
    def test_returns_three_tuple(self, model):
        result = model._select_features(n_features=5)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_train_shape_matches_n_features(self, model):
        X_tr, X_te, feats = model._select_features(n_features=5)
        assert X_tr.shape[1] == 5

    def test_test_shape_matches_n_features(self, model):
        X_tr, X_te, feats = model._select_features(n_features=5)
        assert X_te.shape[1] == 5

    def test_train_row_count_unchanged(self, model):
        X_tr, _, _ = model._select_features(n_features=5)
        assert X_tr.shape[0] == model.X_train.shape[0]

    def test_test_row_count_unchanged(self, model):
        _, X_te, _ = model._select_features(n_features=5)
        assert X_te.shape[0] == model.X_test.shape[0]

    def test_selected_features_length(self, model):
        _, _, feats = model._select_features(n_features=5)
        assert len(feats) == 5

    def test_selected_features_are_column_names(self, model):
        _, _, feats = model._select_features(n_features=5)
        for f in feats:
            assert f in model.X_train.columns

    def test_n_features_equals_total_columns(self, model):
        n = model.X_train.shape[1]
        X_tr, X_te, feats = model._select_features(n_features=n)
        assert X_tr.shape[1] == n


# ──────────────────────────────────────────────
# 6. _random_search_tune
# ──────────────────────────────────────────────

class TestRandomSearchTune:
    def _run_tune(self, model, name, estimator):
        return model._random_search_tune(
            name, estimator, model.X_train, model.y_train
        )

    def test_unknown_name_returns_original_model(self, model):
        lr = LogisticRegression()
        result = self._run_tune(model, "UnknownModel", lr)
        assert result is lr

    def test_zeror_baseline_fits(self, model):
        result = self._run_tune(model, "ZeroR Baseline", DummyClassifier())
        assert hasattr(result, "predict")

    def test_logistic_regression_cv_attrs_attached(self, model):
        result = self._run_tune(model, "Logistic Regression",
                                LogisticRegression(max_iter=200))
        assert hasattr(result, "_cv_best_score")
        assert hasattr(result, "_cv_best_params")

    def test_cv_best_params_is_dict(self, model):
        result = self._run_tune(model, "Logistic Regression",
                                LogisticRegression(max_iter=200))
        assert isinstance(result._cv_best_params, dict)

    def test_cv_best_score_is_float(self, model):
        result = self._run_tune(model, "Logistic Regression",
                                LogisticRegression(max_iter=200))
        assert isinstance(result._cv_best_score, float)

    def test_linearsvc_tunes(self, model):
        result = self._run_tune(model, "LinearSVC",
                                LinearSVC(max_iter=500, random_state=42))
        assert hasattr(result, "predict")

    def test_xgboost_tunes(self, model):
        result = self._run_tune(model, "XGBoost",
                                XGBClassifier(n_estimators=5, random_state=42))
        assert hasattr(result, "predict")

    def test_adaboost_tunes(self, model):
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=5, random_state=42
        )
        result = self._run_tune(model, "AdaBoost", ada)
        assert hasattr(result, "predict")

    def test_bagging_tunes(self, model):
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=3, random_state=42
        )
        result = self._run_tune(model, "Bagging", bag)
        assert hasattr(result, "predict")

    def test_all_known_model_names_no_key_error(self, model):
        """None of the registered names should raise KeyError."""
        pairs = [
            ("ZeroR Baseline", DummyClassifier()),
            ("Logistic Regression", LogisticRegression(max_iter=100)),
            ("LinearSVC", LinearSVC(max_iter=500, random_state=42)),
            ("AdaBoost", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=3, random_state=42)),
            ("XGBoost", XGBClassifier(n_estimators=3, random_state=42)),
            ("Bagging", BaggingClassifier(n_estimators=3, random_state=42)),
        ]
        for name, est in pairs:
            try:
                model._random_search_tune(name, est, model.X_train, model.y_train)
            except KeyError as exc:
                pytest.fail(f"KeyError for '{name}': {exc}")


# ──────────────────────────────────────────────
# 7. _evaluate
# ──────────────────────────────────────────────

class TestEvaluate:
    def _eval(self, model, estimator, name="LR", stage="test"):
        with patch("mlflow.start_run"), \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.set_tag"), \
             patch("mlflow.sklearn.log_model"), \
             patch("src.Model.pickle.dump"):
            return model._evaluate(
                name, estimator,
                model.X_test, model.y_test,
                stage=stage
            )

    def test_returns_dict(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        for k in ("name", "stage", "acc", "precision", "recall",
                  "f2", "roc_auc", "y_prob", "cm", "report"):
            assert k in result, f"Missing key: {k}"

    def test_name_preserved(self, model, fitted_lr):
        result = self._eval(model, fitted_lr, name="MyModel")
        assert result["name"] == "MyModel"

    def test_stage_preserved(self, model, fitted_lr):
        result = self._eval(model, fitted_lr, stage="train")
        assert result["stage"] == "train"

    def test_acc_is_float_in_unit_interval(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert 0.0 <= result["acc"] <= 1.0

    def test_precision_in_unit_interval(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert 0.0 <= result["precision"] <= 1.0

    def test_recall_in_unit_interval(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert 0.0 <= result["recall"] <= 1.0

    def test_f2_in_unit_interval(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert 0.0 <= result["f2"] <= 1.0

    def test_roc_auc_in_unit_interval(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_y_prob_length_matches_test_set(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert len(result["y_prob"]) == len(model.y_test)

    def test_y_prob_values_in_unit_interval(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert np.all((result["y_prob"] >= 0) & (result["y_prob"] <= 1))

    def test_cm_is_2x2(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert result["cm"].shape == (2, 2)

    def test_report_is_string(self, model, fitted_lr):
        result = self._eval(model, fitted_lr)
        assert isinstance(result["report"], str)

    def test_confusion_matrix_plot_created(self, model, fitted_lr):
        self._eval(model, fitted_lr, name="LR", stage="test")
        plots = list(model.plots_dir.glob("*confusion_matrix*.png"))
        assert len(plots) >= 1

    def test_classification_report_file_created(self, model, fitted_lr):
        self._eval(model, fitted_lr, name="LR", stage="test")
        reports = list(model.output_dir.glob("*classification_report*.txt"))
        assert len(reports) >= 1

    def test_linearsvc_sigmoid_probability(self, model):
        """LinearSVC uses decision_function + sigmoid; roc_auc should be valid."""
        svc = _make_mock_estimator(
            "LinearSVC",
            {"C": 1.0, "max_iter": 2000, "random_state": 42},
        )
        result = self._eval(model, svc, name="LinearSVC")
        assert result["roc_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_zeror_baseline_not_logged_to_mlflow_as_model(self, model):
        dummy = _make_mock_estimator("ZeroR Baseline", {"strategy": "most_frequent"})
        with patch("mlflow.start_run"), \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.set_tag"), \
             patch("mlflow.sklearn.log_model") as mock_log_model, \
             patch("src.Model.pickle.dump"):
            model._evaluate("ZeroR Baseline", dummy,
                            model.X_test, model.y_test, stage="test")
            mock_log_model.assert_not_called()

    def test_train_stage_logged(self, model, fitted_lr):
        result = self._eval(model, fitted_lr, stage="train")
        assert result["stage"] == "train"

    def test_logger_called(self, model, fitted_lr):
        with patch.object(model.logger, "info") as mock_info:
            self._eval(model, fitted_lr)
            assert mock_info.call_count > 0


# ──────────────────────────────────────────────
# 8. _plot_roc_curves
# ──────────────────────────────────────────────

class TestPlotRocCurves:
    def _make_results(self, stage="test", n=30):
        y_prob = np.random.uniform(0, 1, n)
        return [_make_result(name="M1", stage=stage, y_prob=y_prob)]

    def test_creates_png_file(self, model):
        results = self._make_results(stage="test")
        y = pd.Series(np.random.randint(0, 2, 30))
        model._plot_roc_curves(results, y, stage="test")
        assert (model.plots_dir / "roc_curves_comparison_test.png").exists()

    def test_filters_by_stage(self, model):
        y = pd.Series(np.random.randint(0, 2, 30))
        results = self._make_results(stage="train") + self._make_results(stage="test")
        # Should only plot stage=test
        model._plot_roc_curves(results, y, stage="test")
        assert (model.plots_dir / "roc_curves_comparison_test.png").exists()

    def test_multiple_models(self, model):
        np.random.seed(1)
        y = pd.Series(np.random.randint(0, 2, 30))
        results = [
            _make_result(name=f"M{i}", stage="test",
                         y_prob=np.random.uniform(0, 1, 30))
            for i in range(3)
        ]
        model._plot_roc_curves(results, y, stage="test")
        assert (model.plots_dir / "roc_curves_comparison_test.png").exists()

    def test_empty_results_for_stage(self, model):
        """When no results match the stage, the method should not raise."""
        y = pd.Series(np.random.randint(0, 2, 30))
        results = self._make_results(stage="train")
        model._plot_roc_curves(results, y, stage="test")   # no test results


# ──────────────────────────────────────────────
# 9. _plot_summary_bar
# ──────────────────────────────────────────────

class TestPlotSummaryBar:
    def _results(self, stage="test"):
        return [
            _make_result(name="A", stage=stage, acc=0.85, roc_auc=0.88, f2=0.80),
            _make_result(name="B", stage=stage, acc=0.78, roc_auc=0.81, f2=0.74),
        ]

    def test_creates_png_file(self, model):
        model._plot_summary_bar(self._results(), stage="test")
        assert (model.plots_dir / "model_comparison_test.png").exists()

    def test_train_stage_file(self, model):
        model._plot_summary_bar(self._results(stage="train"), stage="train")
        assert (model.plots_dir / "model_comparison_train.png").exists()

    def test_none_roc_auc_treated_as_zero(self, model):
        results = [_make_result(name="C", stage="test", roc_auc=None)]
        model._plot_summary_bar(results, stage="test")
        assert (model.plots_dir / "model_comparison_test.png").exists()

    def test_empty_results_no_crash(self, model):
        model._plot_summary_bar([], stage="test")


# ──────────────────────────────────────────────
# 10. _models_vs_baseline
# ──────────────────────────────────────────────

class TestModelsVsBaseline:
    def _results_with_baseline(self, stage="test"):
        return [
            _make_result(name="ZeroR Baseline", stage=stage,
                         acc=0.60, roc_auc=0.50, f2=0.00),
            _make_result(name="LR", stage=stage,
                         acc=0.82, roc_auc=0.87, f2=0.75),
        ]

    def test_creates_csv_file(self, model):
        model._models_vs_baseline(self._results_with_baseline(), stage="test")
        assert (model.output_dir / "models_vs_baseline_test.csv").exists()

    def test_csv_has_correct_columns(self, model):
        model._models_vs_baseline(self._results_with_baseline(), stage="test")
        df = pd.read_csv(model.output_dir / "models_vs_baseline_test.csv")
        for col in ("Model", "Delta Accuracy", "Delta ROC-AUC", "Delta F2-Score"):
            assert col in df.columns

    def test_csv_excludes_baseline_row(self, model):
        model._models_vs_baseline(self._results_with_baseline(), stage="test")
        df = pd.read_csv(model.output_dir / "models_vs_baseline_test.csv")
        assert "ZeroR Baseline" not in df["Model"].values

    def test_delta_accuracy_correct(self, model):
        model._models_vs_baseline(self._results_with_baseline(), stage="test")
        df = pd.read_csv(model.output_dir / "models_vs_baseline_test.csv")
        row = df[df["Model"] == "LR"].iloc[0]
        assert pytest.approx(row["Delta Accuracy"], abs=1e-4) == 0.82 - 0.60

    def test_no_baseline_logs_warning(self, model):
        results = [_make_result(name="LR", stage="test")]
        with patch.object(model.logger, "warning") as mock_warn:
            model._models_vs_baseline(results, stage="test")
            mock_warn.assert_called_once()

    def test_no_baseline_no_csv_created(self, model):
        results = [_make_result(name="LR", stage="test")]
        model._models_vs_baseline(results, stage="test")
        assert not (model.output_dir / "models_vs_baseline_test.csv").exists()

    def test_none_roc_auc_yields_nan_delta(self, model):
        results = [
            _make_result(name="ZeroR Baseline", stage="test",
                         acc=0.5, roc_auc=None, f2=0.0),
            _make_result(name="LR", stage="test",
                         acc=0.8, roc_auc=None, f2=0.7),
        ]
        model._models_vs_baseline(results, stage="test")
        df = pd.read_csv(model.output_dir / "models_vs_baseline_test.csv")
        assert np.isnan(df.loc[0, "Delta ROC-AUC"])

    def test_filters_by_stage(self, model):
        results = self._results_with_baseline(stage="train") + \
                  self._results_with_baseline(stage="test")
        model._models_vs_baseline(results, stage="train")
        # Only train CSV should exist
        assert (model.output_dir / "models_vs_baseline_train.csv").exists()


# ──────────────────────────────────────────────
# 11. _save_results_csv
# ──────────────────────────────────────────────

class TestSaveResultsCsv:
    def _results(self):
        return [
            _make_result(name="A", stage="train", acc=0.9,
                         prec=0.88, rec=0.85, f2=0.86, roc_auc=0.91),
            _make_result(name="A", stage="test",  acc=0.82,
                         prec=0.80, rec=0.78, f2=0.79, roc_auc=0.85),
        ]

    def test_csv_file_created(self, model):
        model._save_results_csv(self._results())
        assert (model.output_dir / "model_scores.csv").exists()

    def test_csv_has_required_columns(self, model):
        model._save_results_csv(self._results())
        df = pd.read_csv(model.output_dir / "model_scores.csv")
        for col in ("Model", "Stage", "Accuracy", "ROC-AUC",
                    "Precision", "Recall", "F2-Score"):
            assert col in df.columns

    def test_csv_row_count_matches_results(self, model):
        results = self._results()
        model._save_results_csv(results)
        df = pd.read_csv(model.output_dir / "model_scores.csv")
        assert len(df) == len(results)

    def test_accuracy_value_preserved(self, model):
        model._save_results_csv(self._results())
        df = pd.read_csv(model.output_dir / "model_scores.csv")
        train_row = df[df["Stage"] == "train"].iloc[0]
        assert pytest.approx(train_row["Accuracy"], abs=1e-4) == 0.9

    def test_empty_results_creates_empty_csv(self, model):
        model._save_results_csv([])
        csv_path = model.output_dir / "model_scores.csv"
        assert csv_path.exists()
        # An empty results list produces a DataFrame with no rows and no
        # columns → pandas writes a file containing only a newline.
        assert csv_path.stat().st_size > 0   # file is not missing


class TestRun:
    @pytest.fixture
    def run_results(self, model):
        with patch("mlflow.start_run"), \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_artifact"), \
             patch("mlflow.set_tag"), \
             patch("mlflow.sklearn.log_model"), \
             patch("src.Model.pickle.dump"):
            return model.run()

    def test_returns_list(self, run_results):
        assert isinstance(run_results, list)

    def test_result_count_is_models_times_two_stages(self, run_results):
        # 6 models × 2 stages (train + test) = 12
        assert len(run_results) == 12

    def test_every_result_has_name(self, run_results):
        for r in run_results:
            assert isinstance(r["name"], str) and r["name"]

    def test_stages_are_train_and_test(self, run_results):
        stages = {r["stage"] for r in run_results}
        assert stages == {"train", "test"}

    def test_model_scores_csv_created(self, model, run_results):
        assert (model.output_dir / "model_scores.csv").exists()

    def test_roc_curve_plots_created(self, model, run_results):
        assert (model.plots_dir / "roc_curves_comparison_test.png").exists()
        assert (model.plots_dir / "roc_curves_comparison_train.png").exists()

    def test_summary_bar_plots_created(self, model, run_results):
        assert (model.plots_dir / "model_comparison_test.png").exists()
        assert (model.plots_dir / "model_comparison_train.png").exists()

    def test_models_vs_baseline_csv_created(self, model, run_results):
        assert (model.output_dir / "models_vs_baseline_test.csv").exists()
        assert (model.output_dir / "models_vs_baseline_train.csv").exists()

    def test_all_six_model_names_present(self, run_results):
        names = {r["name"] for r in run_results}
        for expected in ("ZeroR Baseline", "Logistic Regression",
                         "LinearSVC", "AdaBoost", "XGBoost", "Bagging"):
            assert expected in names

    def test_accuracy_values_in_unit_interval(self, run_results):
        for r in run_results:
            assert 0.0 <= r["acc"] <= 1.0