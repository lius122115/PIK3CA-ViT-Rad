"""Train radiomic, ViT-derived, and integrated models without validation leakage."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, roc_curve
from pipeline import select_model, fit_final


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def youden_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(thresholds[np.argmax(tpr - fpr)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["data"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    c = cfg["columns"]; clinical = pd.read_csv(cfg["data"]["clinical_table"])
    rad = pd.read_csv(cfg["data"]["radiomics_table"]); vit = pd.read_csv(cfg["data"]["vit_table"])
    data = clinical[[c["patient_id"], c["label"], c["cohort"]]].merge(rad, on=c["patient_id"]).merge(vit, on=c["patient_id"])
    data[c["label"]] = data[c["label"]].astype(int)
    train = data[data[c["cohort"]] == "cohort1"].copy()
    external = data[data[c["cohort"]] == "cohort2"].copy()
    cohort3 = data[data[c["cohort"]] == "cohort3"].copy()
    tr, te = train_test_split(train, test_size=cfg["split"]["test_size"], random_state=cfg["seed"], stratify=train[c["label"]])
    target = c["label"]; pid = c["patient_id"]
    rad_cols = [x for x in rad.columns if x != pid]
    vit_cols = [x for x in vit.columns if x != pid]
    feature_sets = {"radiomics": rad_cols, "vit": vit_cols, "integrated": rad_cols + vit_cols}
    if not cfg["model"].get("xgb_candidates"):
        raise ValueError("config.yaml must contain the exact XGBoost settings used in the final analysis.")
    radiomics_candidates = [("logistic_regression", {}), ("random_forest", {}), ("svm", {}), ("xgboost", cfg["model"]["xgb_candidates"][0])]
    xgb_candidates = [("xgboost", x) for x in cfg["model"]["xgb_candidates"]]
    all_rows = []; selections = {}
    for set_name, cols in feature_sets.items():
        candidates = radiomics_candidates if set_name == "radiomics" else xgb_candidates
        best, cv = select_model(tr[cols], tr[target], candidates, cfg)
        cv.to_csv(out / f"{set_name}_training_cv.csv", index=False)
        selector, model = fit_final(tr[cols], tr[target], best["model"], best["params"], cfg)
        expected_counts = cfg.get("expected_final_feature_counts", {})
        expected = expected_counts.get(set_name)
        if expected is not None and len(selector.features) != expected:
            raise RuntimeError(
                f"{set_name} model retained {len(selector.features)} features; "
                f"the 7.26 Supplementary Materials specify {expected}. "
                "Check the exact final data preprocessing and feature-selection settings."
            )
        train_probability = model.predict_proba(selector.transform(tr[cols]))[:, 1]
        threshold = youden_threshold(tr[target].to_numpy(), train_probability)
        selections[set_name] = {"model": best["model"], "params": best["params"], "features": selector.features, "cv_auc": best["cv_auc"], "youden_threshold": threshold}
        for split_name, frame in [("training", tr), ("internal_test", te), ("external_validation", external), ("cohort3", cohort3)]:
            if len(frame) == 0: continue
            p = model.predict_proba(selector.transform(frame[cols]))[:, 1]
            row = {"feature_set": set_name, "split": split_name, "threshold": threshold, **metrics(frame[target].to_numpy(), p, threshold)}
            all_rows.append(row)
            pd.DataFrame({pid: frame[pid].to_numpy(), target: frame[target].to_numpy(), "probability": p}).to_csv(out / f"{set_name}_{split_name}_predictions.csv", index=False)
    (out / "final_models.json").write_text(json.dumps(selections, indent=2, default=str))
    pd.DataFrame(all_rows).to_csv(out / "performance_summary.csv", index=False)


if __name__ == "__main__": main()
