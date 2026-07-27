"""Leakage-controlled feature selection and model fitting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def find_correlated(X: pd.DataFrame, cutoff: float) -> list[str]:
    """Approximate caret::findCorrelation: remove the feature with the
    greater mean absolute correlation for each remaining high-correlation pair."""
    corr = X.corr(method="pearson").abs().fillna(0.0)
    removed = []
    while corr.shape[0] > 1:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        hits = np.argwhere(upper.to_numpy() > cutoff)
        if len(hits) == 0:
            break
        i, j = hits[0]
        names = list(corr.columns)
        left, right = names[i], names[j]
        mean_left = (corr.loc[left].sum() - 1.0) / max(len(names) - 1, 1)
        mean_right = (corr.loc[right].sum() - 1.0) / max(len(names) - 1, 1)
        drop = left if mean_left >= mean_right else right
        removed.append(drop)
        corr = corr.drop(index=drop, columns=drop)
    return removed


@dataclass
class Selector:
    imputer: Any
    scaler: Any
    features: list[str]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.features)
        return self.scaler.transform(self.imputer.transform(X))


def fit_selector(X: pd.DataFrame, y: pd.Series, p_threshold=.05, corr_cutoff=.75, lasso_cv=5, lasso_max_iter=20000, seed=42) -> Selector:
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric candidate features remain.")
    zero = X.columns[X.nunique(dropna=False) <= 1]
    X = X.drop(columns=zero)
    if X.shape[1] == 0:
        raise ValueError("All candidate features have zero variance in this training fold.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xi = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    Xs = pd.DataFrame(scaler.fit_transform(Xi), columns=X.columns, index=X.index)
    y = pd.Series(y, index=X.index).astype(int)
    p = {}
    for col in Xs.columns:
        a, b = Xs.loc[y == 0, col], Xs.loc[y == 1, col]
        p_t = ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue
        p_w = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) and len(b) else 1.0
        p[col] = min(float(p_t), float(p_w))
    screened = [c for c, value in p.items() if value < p_threshold]
    if not screened:
        raise ValueError("No feature passed the t-test/Wilcoxon screening threshold in this fold.")
    keep = [c for c in screened if c not in find_correlated(Xs[screened], corr_cutoff)]
    if not keep:
        raise ValueError("Correlation filtering removed all screened features.")
    lasso = LogisticRegressionCV(
        Cs=20, cv=lasso_cv, penalty="l1", solver="liblinear", scoring="roc_auc",
        max_iter=lasso_max_iter, random_state=seed, refit=True
    )
    lasso.fit(Xs[keep], y)
    selected = [c for c, coef in zip(keep, lasso.coef_[0]) if abs(coef) > 1e-12]
    if not selected:
        raise ValueError("LASSO selected no nonzero coefficient features.")
    # Refit imputation/scaling on the same fold using the selected columns so the
    # frozen object exactly describes the columns and parameters applied downstream.
    final_imputer = SimpleImputer(strategy="median").fit(X[selected])
    final_scaler = StandardScaler().fit(final_imputer.transform(X[selected]))
    return Selector(final_imputer, final_scaler, selected)


def make_model(name: str, params: dict, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=10000, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1, **params)
    if name == "svm":
        return SVC(probability=True, random_state=seed, **params)
    if name == "xgboost":
        forbidden = {"scale_pos_weight", "max_delta_step"}
        supplied_forbidden = sorted(forbidden.intersection(params))
        if supplied_forbidden:
            raise ValueError(
                "The final analysis used no class-weight adjustment or oversampling; "
                f"remove these XGBoost parameters: {supplied_forbidden}."
            )
        xgb_params = dict(params)
        xgb_params.setdefault("objective", "binary:logistic")
        xgb_params.setdefault("eval_metric", "auc")
        return XGBClassifier(random_state=seed, n_jobs=1, **xgb_params)
    raise ValueError(name)


def nested_cv_score(X: pd.DataFrame, y: pd.Series, model_name: str, params: dict, cfg: dict) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    scores = []
    for train_idx, valid_idx in skf.split(X, y):
        selector = fit_selector(X.iloc[train_idx], y.iloc[train_idx], **cfg["selection"], seed=cfg["seed"])
        model = make_model(model_name, params, cfg["seed"])
        model.fit(selector.transform(X.iloc[train_idx]), y.iloc[train_idx])
        p = model.predict_proba(selector.transform(X.iloc[valid_idx]))[:, 1]
        scores.append(roc_auc_score(y.iloc[valid_idx], p))
    return float(np.mean(scores))


def fit_final(X: pd.DataFrame, y: pd.Series, model_name: str, params: dict, cfg: dict):
    selector = fit_selector(X, y, **cfg["selection"], seed=cfg["seed"])
    model = make_model(model_name, params, cfg["seed"])
    model.fit(selector.transform(X), y)
    return selector, model


def select_model(X: pd.DataFrame, y: pd.Series, model_candidates: list[tuple[str, dict]], cfg: dict):
    results = []
    for name, params in model_candidates:
        score = nested_cv_score(X, y, name, params, cfg)
        results.append({"model": name, "params": params, "cv_auc": score})
    best = max(results, key=lambda z: z["cv_auc"])
    return best, pd.DataFrame(results)
