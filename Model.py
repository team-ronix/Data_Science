import os
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import pickle
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, auc,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, fbeta_score
)


class Model:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        business_statistics: pd.DataFrame,
        output_dir: str = "model_outputs",
        random_search_iter: int = 1,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.business_statistics = business_statistics

        self.random_search_iter = random_search_iter

        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup MLflow
        db_path = Path(__file__).parent / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        self.mlflow_experiment_name = "Loan_Status_Classification"
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        self._setup_logging()

    def _setup_logging(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/ModelDevelopment.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("ModelDevelopment")
        self.logger.info("Logger initialized successfully.")

    def _extract_hyperparameters(self, name: str, model) -> dict:
        params = {}
        
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            # Filter to only include main hyperparameters (exclude estimator__ nested params for clarity)
            for key, value in model_params.items():
                if not key.startswith('estimator__'):
                    try:
                        # Convert to string to ensure MLflow compatibility
                        params[key] = str(value)
                    except:
                        pass
        
        # Add custom CV best params if available (from hyperparameter tuning)
        if hasattr(model, '_cv_best_params'):
            for key, value in model._cv_best_params.items():
                try:
                    params[f"best_{key}"] = str(value)
                except:
                    pass
        
        return params

    def _compute_business_metrics(self, cm: np.ndarray) -> dict:
        """
        Compute business-related metrics for the lending club context.
        Assumption: Class 0 = Charge Off (bad loan), Class 1 = Fully Paid (good loan)
        """
        tn, fp, fn, tp = cm.ravel()
        
        # Business metrics
        avg_loan_profit = self.business_statistics["avg_loan_profit"].iloc[0]
        avg_loan_amount = self.business_statistics["avg_loan_amount"].iloc[0]
        
        # average of missed profit from incorrectly rejected loans (false positives)
        avg_missed_profit = fp * avg_loan_profit
        
        # average of gained profit from correctly approved loans (true negatives)
        avg_gained_profit = tn * avg_loan_profit
        
        # average of lost loan amounts from incorrectly approved bad loans (false negatives)
        avg_lost_loans_amnts = fn * avg_loan_amount
        
        # average of saved loss from correctly rejected bad loans (true positives)
        avg_saved_loans_amnts = tp * avg_loan_amount
                
        return {
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,  # Risk of approving bad loans
            "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,  # Rate of correctly identifying good loans
            "avg_missed_profit": avg_missed_profit,
            "avg_gained_profit": avg_gained_profit,
            "avg_lost_loans_amount": avg_lost_loans_amnts,
            "avg_saved_loans_amnts": avg_saved_loans_amnts,
        }

    def _save_matplotlib_plot(self, file_name: str) -> None:
        plt.tight_layout()
        plt.savefig(self.plots_dir / file_name, dpi=160, bbox_inches="tight")
        plt.close()

    def _select_features(self, n_features: int = 30) -> pd.DataFrame:
        self.logger.info("Start selecting top features based on correlation with target.")
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_kbest = selector.fit_transform(self.X_train, self.y_train)
        X_test_kbest = selector.transform(self.X_test)
        return X_train_kbest, X_test_kbest

    def _random_search_tune(self, name: str, model, X_train, y_train):
        scale_pos_weight = (y_train == 0).sum() / ((y_train == 1).sum() + 1e-6)
        param_distributions = {
            "ZeroR Baseline": {
                "strategy": ["most_frequent", "prior", "stratified", "uniform"],
            },
            "Logistic Regression": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [200, 300, 500, 1000],
            },
            "AdaBoost": {
                "n_estimators": [100, 200, 300, 400],
                "learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                "estimator__max_depth": [1, 2, 3],
                "estimator__min_samples_leaf": [1, 5, 10, 20],
            },
            "LinearSVC": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "tol": [1e-4, 1e-3, 1e-2],
                "max_iter": [1000, 2000, 5000],
            },
            "XGBoost": {
                "n_estimators": [100, 200, 300, 400],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.3],
                "scale_pos_weight": [
                    scale_pos_weight,
                    scale_pos_weight * 0.5,
                    scale_pos_weight * 1.5,
                ],
            },
            "Bagging": {
                "n_estimators": [100, 200, 300],
                "max_samples": [0.6, 0.8, 1.0],
                "max_features": [0.6, 0.8, 1.0],
                "bootstrap": [True],
                "bootstrap_features": [False, True],
            }
        }

        if name not in param_distributions:
            self.logger.info(f"No hyperparameters defined for {name}.")
            return model

        # We use f2 score to give more weight to recall than precision
        self.logger.info(f"Starting RandomizedSearchCV for {name}")
        tuner = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=self.random_search_iter,
            scoring = make_scorer(fbeta_score, beta=2, pos_label=1),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        tuner.fit(X_train, y_train)
        self.logger.info(f"{name} best CV F2 (class 1): {tuner.best_score_:.4f}")
        self.logger.info(f"{name} best params: {tuner.best_params_}")
        best = tuner.best_estimator_
        best._cv_best_score = tuner.best_score_
        best._cv_best_params = tuner.best_params_
        return best

    def _evaluate(self, name: str, model, X, y, stage: str = "test") -> dict:
        self.logger.info(f"Evaluating {name}")
        y_pred = model.predict(X)
        roc_auc = None
        if name == "LinearSVC":
            y_score = model.decision_function(X)
            y_prob = 1 / (1 + np.exp(-y_score))
            roc_auc = roc_auc_score(y, y_prob)
        else:   
            y_prob = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_prob)

        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f2s = fbeta_score(y, y_pred, beta=2, zero_division=0)  # F2 score
        report = classification_report(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        logged_roc_auc = 'N/A'
        if roc_auc is not None:
            logged_roc_auc = f"{roc_auc:.4f}"
        self.logger.info(f"{name}/{stage} - Accuracy: {acc:.4f} | ROC-AUC: {logged_roc_auc}")
        self.logger.info(f"\n{report}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Fully Paid", "Charge Off"]).plot(
            ax=ax, colorbar=True, cmap="Blues"
        )
        ax.set_title(f"{name} - Confusion Matrix")
        self._save_matplotlib_plot(f"{name.replace(' ', '_')}_{stage}_confusion_matrix.png")

        # ===== MLflow Logging =====
        with mlflow.start_run(run_name=f"{name} ({stage})"):
            # Log model name
            mlflow.set_tag("model_name", name)
            
            # Extract and log hyperparameters
            params = self._extract_hyperparameters(name, model)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log standard metrics
            mlflow.log_metric(f"accuracy_{stage}", acc)
            mlflow.log_metric(f"precision_{stage}", precision)
            mlflow.log_metric(f"recall_{stage}", recall)
            mlflow.log_metric(f"f2_score_{stage}", f2s)
            if roc_auc is not None:
                mlflow.log_metric(f"roc_auc_{stage}", roc_auc)
            
            # Log business metrics
            business_metrics = self._compute_business_metrics(cm)
            mlflow.log_metric(f"false_positive_rate_{stage}", business_metrics["false_positive_rate"])
            mlflow.log_metric(f"true_positive_rate_{stage}", business_metrics["true_positive_rate"])
            mlflow.log_metric(f"avg_missed_profit_{stage}", business_metrics["avg_missed_profit"])
            mlflow.log_metric(f"avg_gained_profit_{stage}", business_metrics["avg_gained_profit"])
            mlflow.log_metric(f"avg_lost_loans_amount_{stage}", business_metrics["avg_lost_loans_amount"])
            mlflow.log_metric(f"avg_saved_loans_amnts_{stage}", business_metrics["avg_saved_loans_amnts"])

            # Log confusion matrix values
            tn, fp, fn, tp = cm.ravel()
            mlflow.log_metric(f"true_negatives_{stage}", int(tn))
            mlflow.log_metric(f"false_positives_{stage}", int(fp))
            mlflow.log_metric(f"false_negatives_{stage}", int(fn))
            mlflow.log_metric(f"true_positives_{stage}", int(tp))
            
            # Log classification report as artifact
            report_file = self.output_dir / f"{name.replace(' ', '_')}_{stage}_classification_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            mlflow.log_artifact(str(report_file))
            
            # Log confusion matrix plot
            cm_plot_file = self.plots_dir / f"{name.replace(' ', '_')}_{stage}_confusion_matrix.png"
            if cm_plot_file.exists():
                mlflow.log_artifact(str(cm_plot_file))
            
            # Save model as artifact (except for DummyClassifier)
            if name != "ZeroR Baseline":
                try:
                    mlflow.sklearn.log_model(model, f"{name.replace(' ', '_')}_model")
                except:
                    # Fallback for non-sklearn models
                    model_file = self.output_dir / f"{name.replace(' ', '_')}_model.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(str(model_file))

        return {
            "name": name, "model": model, "stage": stage,
            "acc": acc, "precision": precision, "recall": recall, "f2": f2s,
            "roc_auc": roc_auc, "y_prob": y_prob, "report": report, "cm": cm,
        }

    def _plot_roc_curves(self, results: list, y, stage: str = "test") -> None:
        filtered_results = [r for r in results if r["stage"] == stage]
        plt.figure(figsize=(10, 8))
        for r in filtered_results:
            y_prob = r["y_prob"]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.8, alpha=0.8, label=f"{r['name']} {r['stage']} (AUC={roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - Binary Classification (All Models) ({stage})")
        plt.legend(loc="lower right", fontsize=8)
        self._save_matplotlib_plot(f"roc_curves_comparison_{stage}.png")
        self.logger.info("ROC curves (binary) saved.")

    def _plot_summary_bar(self, results: list, stage: str = "test") -> None:
        filtered_results = [r for r in results if r["stage"] == stage]
        names = [r["name"] for r in filtered_results]
        accs = [r["acc"] for r in filtered_results]
        aucs = [r["roc_auc"] if r["roc_auc"] is not None else 0.0 for r in filtered_results]
        f2s = [r["f2"] for r in filtered_results]
        x, w = np.arange(len(names)), 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w, accs, w, label="Accuracy", color="#4C8BE2")
        ax.bar(x + w, aucs, w, label="ROC-AUC", color="#F4A261")
        ax.bar(x, f2s, w, label="F2-Score", color="#2A9D8F")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Model Comparison - Accuracy vs ROC-AUC ({stage})")
        ax.legend()
        self._save_matplotlib_plot(f"model_comparison_{stage}.png")
        self.logger.info("Model comparison bar chart saved.")

    def _models_vs_baseline(self, results: list, stage: str = "test") -> None:
        filtered_results = [r for r in results if r["stage"] == stage]
        baseline = next((r for r in filtered_results if r["name"] == "ZeroR Baseline"), None)
        if baseline is None:
            self.logger.warning("No baseline model found; skipping baseline context.")
            return

        self.logger.info(f"Models vs {baseline['name']} ({stage}):")
        rows = []
        for r in filtered_results:
            if r["name"] == baseline["name"]:
                continue
            d_auc = (
                (r["roc_auc"] - baseline["roc_auc"])
                if r["roc_auc"] is not None and baseline["roc_auc"] is not None
                else float("nan")
            )
            self.logger.info(
                f"{r['name']} ({stage}): "
                f"dAcc={r['acc'] - baseline['acc']:+.4f}, "
                f"dROC-AUC={d_auc:+.4f}, "
                f"dF2={r['f2'] - baseline['f2']:+.4f}"
            )
            rows.append({
                "Model": r["name"],
                "Delta Accuracy": r["acc"] - baseline["acc"],
                "Delta ROC-AUC": d_auc,
                "Delta F2-Score": r["f2"] - baseline["f2"],
            })
        pd.DataFrame(rows).to_csv(self.output_dir / f"models_vs_baseline_{stage}.csv", index=False)
        self.logger.info(f"Models vs baseline saved to {self.output_dir}/models_vs_baseline_{stage}.csv")

    def _save_results_csv(self, results: list) -> None:
        rows = [{
            "Model": r["name"], "Stage": r["stage"], "Accuracy": r["acc"], "ROC-AUC": r["roc_auc"],
            "Precision": r["precision"], "Recall": r["recall"], "F2-Score": r["f2"]
        } for r in results]
        pd.DataFrame(rows).to_csv(self.output_dir / "model_scores.csv", index=False)
        self.logger.info(f"Scores saved to {self.output_dir}/model_scores.csv")
 
    def run(self) -> list:
        X_train_selected, X_test_selected = self._select_features(n_features=30)
        models = {
            "ZeroR Baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(
                max_iter=500, random_state=42, class_weight="balanced"
            ),
            "LinearSVC": LinearSVC(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
            # "AdaBoost": AdaBoostClassifier(
            #     estimator=DecisionTreeClassifier(
            #         max_depth=2, min_samples_leaf=10, random_state=42, class_weight="balanced"
            #     ),
            #     n_estimators=300, learning_rate=0.5, random_state=42
            # ),
            # "XGBoost": XGBClassifier(
            #     n_estimators=300,
            #     max_depth=6,
            #     learning_rate=0.05,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=42,
            #     eval_metric="logloss",
            # ),
            # "Bagging": BaggingClassifier(
            #     estimator=DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            #     n_estimators=200,
            #     max_samples=0.8,
            #     max_features=0.8,
            #     bootstrap=True,
            #     random_state=42,
            #     n_jobs=-1,
            # ),
        }

        tuned_models = {}
        for name, model in models.items():
            tuned_models[name] = self._random_search_tune(name, model, X_train_selected, self.y_train)

        results = []
        for name, model in tuned_models.items():
            r = self._evaluate(name, model, X_train_selected, self.y_train, stage="train")
            results.append(r)
            r = self._evaluate(name, model, X_test_selected, self.y_test, stage="test")
            results.append(r)

        self._plot_roc_curves(results, self.y_train, stage="train")
        self._plot_roc_curves(results, self.y_test, stage="test")
        
        self._plot_summary_bar(results, stage="train")
        self._plot_summary_bar(results, stage="test")
        
        self._models_vs_baseline(results, stage="train")
        self._models_vs_baseline(results, stage="test")

        self._save_results_csv(results)
        return results


if __name__ == "__main__":
    data_folder = Path("data")
    train = pd.read_csv(data_folder / "train_norm.csv")
    test = pd.read_csv(data_folder / "test.csv")
    business_statistics = pd.read_csv(data_folder / "business_statistics.csv")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    model_dev = Model(
        X_train=train.drop(columns=["loan_status"]),
        y_train=train["loan_status"],
        X_test=test.drop(columns=["loan_status"]),
        y_test=test["loan_status"],
        business_statistics=business_statistics,
        output_dir="model_outputs",
        random_search_iter=1,
    )
    results = model_dev.run()