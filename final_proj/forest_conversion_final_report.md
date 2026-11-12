# Forest conversion on unseen parcels, 2015–2025

The source contains 1000 parcels and 12 columns. Of 285 forest parcels, 230 stayed forest, 29 became cropland, 16 became built-up, eight became grassland and two became barren. The first three destinations form the 275 training labels. The ten remaining parcels appear in the risk file with their observed destination and outside-scope flag.

## Predictors and assumptions

The pipeline uses dist_to_city_km, dist_to_road_km, slope_deg, rainfall_mm, soil_quality_idx, ndvi_2015, soil_suitability, road_accessibility, protected_area. Soil suitability multiplies soil quality by (1 − slope/45) and clips it to [0,1]. Road accessibility equals 1/(1 + road distance in km). These proxies describe soil and access. The soil score is an unvalidated heuristic. Undated distance, protection and rainfall records are assumed available in 2015 and soil and slope sufficiently stable. Population growth has an unknown measurement period. It and the outcome flag were excluded.

## Model comparison

Four stratified folds within the 220-parcel training partition selected **XGBoost** by macro-F1. The remaining 55 parcels supplied the test results below. CV variation is the sample standard deviation across folds.

| model | CV_macro_F1_mean | CV_macro_F1_std | accuracy | balanced_accuracy | macro_F1 | weighted_F1 | macro_OVR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.394 | 0.055 | 0.545 | 0.370 | 0.329 | 0.607 | 0.765 |
| Random Forest | 0.373 | 0.103 | 0.782 | 0.560 | 0.530 | 0.795 | 0.790 |
| Logistic Regression | 0.362 | 0.113 | 0.473 | 0.541 | 0.366 | 0.555 | 0.719 |
| Majority baseline | 0.304 | 0.000 | 0.836 | 0.333 | 0.304 | 0.762 | 0.500 |

The selected model reached **0.329 test macro-F1** versus 0.304 for always predicting forest persistence. Its accuracy was 54.5% versus 83.6% for the majority rule. The previous workflow already inspected this holdout. No test-driven tuning was performed. The requested forest depth cap was fixed before this run.

| outcome | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| no_change | 0.824 | 0.609 | 0.700 | 46.000 |
| to_agricultural | 0.077 | 0.167 | 0.105 | 6.000 |
| to_built_up | 0.125 | 0.333 | 0.182 | 3.000 |

Built-up recall rests on 3 test cases. One additional correct prediction changes it by 33%.

## Explanation of the full model

protected_area_0, dist_to_road_km, dist_to_city_km had the largest mean absolute SHAP contributions to the combined conversion score. Indicator suffixes on protected_area denote one-hot categories. Positive SHAP contributions increase a parcel's score relative to the training background. These associations cannot establish causation. The permutation chart measures held-out performance of the evaluation fit and may divide importance among correlated predictors.

## Retrospective parcel ranking

The five-fold ranking scores each eligible parcel while excluding it from fitting. The separately saved pipeline trained on all 275 eligible parcels scores the ten outside-scope parcels under the restricted outcome definition. Among 285 parcels, 64 are High (at least 0.66), 129 Medium (0.33 to below 0.66) and 92 Low (below 0.33).

| region | risk_score | most_likely_conversion_type | most_likely_overall_outcome | scope_status |
| --- | --- | --- | --- | --- |
| 312 | 0.941 | to_agricultural | to_agricultural | in_scope |
| 134 | 0.918 | to_agricultural | to_agricultural | in_scope |
| 111 | 0.894 | to_built_up | to_built_up | in_scope |
| 438 | 0.888 | to_built_up | to_built_up | in_scope |
| 344 | 0.885 | to_built_up | to_built_up | in_scope |
| 695 | 0.884 | to_agricultural | to_agricultural | in_scope |
| 449 | 0.883 | to_built_up | to_built_up | in_scope |
| 127 | 0.866 | to_built_up | to_built_up | in_scope |
| 189 | 0.862 | to_built_up | to_built_up | in_scope |
| 931 | 0.861 | to_built_up | to_built_up | in_scope |

## Limits and future improvements

The scores are uncalibrated screening indicators. The parcel split measures performance within one decade. Spatial and future-year performance are unknown. Built-up combines residential and industrial conversion, and these labels do not measure sustainability. With this CSV, repeated nested stratified splits could assess split sensitivity. Calibration would require separate validation, particularly for the rare outcomes.
