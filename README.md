# Reproducible analysis code

This repository contains the analysis workflow for the DCE MRI Vision Transformer–radiomics (ViT-Rad) study. Patient data, images, masks, clinical labels, metabolomic/lipidomic tables, and trained weights are not included because the institutional data are restricted.

## Workflow

1. Extract radiomic features from the preprocessed, whole-tumor DCE MRI volumes with PyRadiomics.
2. Extract 768-dimensional class-token features from an ImageNet-pretrained ViT-Base-Patch16 model using the largest tumor-containing axial section.
3. Split Cohort 1 at the patient level into training and internal test sets with stratification by PIK3CA status.
4. For each of the radiomic, ViT-derived, and integrated candidate feature sets, perform all fitted operations inside each training fold: median imputation, z-score standardization, univariable Student *t* and Wilcoxon rank-sum screening, correlation filtering (absolute Pearson correlation >0.75), and LASSO logistic selection.
5. Evaluate candidate classifiers and XGBoost settings using fivefold stratified cross-validation within the training set only. No internal-test, external-validation, or Cohort 3 data are used for feature selection, model selection, or hyperparameter tuning.
6. Refit the complete selected pipeline on the full training set and apply it unchanged to the internal test set, external validation cohort, and Cohort 3.

The integrated feature set is selected jointly from the combined radiomic and ViT-derived candidates; it is not the union of the separately selected radiomic and ViT feature sets. The final 7.26 analysis retained 8 radiomic features, 10 ViT-derived features, and 15 integrated features; the code checks these counts. The final integrated model probability is the ViT-Rad score.

## Run

```bash
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
# Edit config.yaml and provide local, deidentified data paths.
python feature_extraction/radiomics_extract.py --config config.yaml \
  --image-dir /path/to/preprocessed_images --mask-dir /path/to/tumor_masks
python feature_extraction/vit_extract.py --config config.yaml \
  --image-dir /path/to/preprocessed_images --mask-dir /path/to/tumor_masks
python model_development/run_models.py --config config.yaml
```

The scripts write only feature tables, fitted-pipeline metadata, predictions, and evaluation summaries. They do not write patient identifiers to figures or upload data. Feature-attribution analyses were performed separately and are not included in this code package.

## Data contract

The clinical table must contain a unique `patient_id`, a binary `pik3ca_status` column (1 = mutated, 0 = wild type), and optional cohort columns (`cohort`, with values `cohort1`, `cohort2`, and `cohort3`). Feature tables must contain the same `patient_id` values and one column per candidate feature. Image and mask files must share the patient identifier in their filenames.

## Reproducibility notes

No class-weight or oversampling procedure was used. The observed PIK3CA-mutated proportion was approximately 36%, and model fitting used the original training-fold data. The final XGBoost setting used 12 boosting rounds; this number was selected by fivefold stratified cross-validation within the training set according to the validation-fold AUC. The internal test set, external validation cohort, and Cohort 3 were used only for prediction and evaluation.

Model selection must be described as based on cross-validation within the training set. Any Supplementary Materials sentence stating that XGBoost was selected according to external-validation AUC must be corrected before upload because it would imply information leakage.
