# Forest conversion on unseen parcels, 2015–2025

The source contains 1000 parcels and 12 columns. Of 285 forest parcels, 230 stayed forest, 29 became cropland, 16 became built-up, eight became grassland and two became barren. The first three destinations form the 275 training labels. The ten remaining parcels appear in the risk file with their observed destination and outside-scope flag.

## Predictors and assumptions

The pipeline uses dist_to_city_km, dist_to_road_km, slope_deg, rainfall_mm, soil_quality_idx, ndvi_2015, soil_suitability, road_accessibility, protected_area. Soil suitability multiplies soil quality by (1 − slope/45) and clips it to [0,1]. Road accessibility equals 1/(1 + road distance in km). These proxies describe soil and access. The soil score is an unvalidated heuristic. Undated distance, protection and rainfall records are assumed available in 2015 and soil and slope sufficiently stable. Population growth has an unknown measurement period. It and the outcome flag were excluded.

## Model comparison

Four stratified folds within the 220-parcel training partition selected **Boost depth=1, weight=0.5** using the training accuracy and minority-detection constraints described in the notebook, then macro-F1. The remaining 55 parcels supplied the test results below. CV variation is the sample standard deviation across folds.

| model | CV_macro_F1_mean | CV_macro_F1_std | accuracy | balanced_accuracy | macro_F1 | weighted_F1 | macro_OVR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | 0.304 | 0.000 | 0.836 | 0.333 | 0.304 | 0.762 | 0.500 |
| Logistic C=1.0, weight=0.5 | 0.350 | 0.075 | 0.727 | 0.394 | 0.366 | 0.723 | 0.738 |
| Forest depth=4, weight=0.75 | 0.361 | 0.082 | 0.782 | 0.415 | 0.406 | 0.758 | 0.745 |
| Boost depth=1, weight=0.5 | 0.339 | 0.056 | 0.818 | 0.326 | 0.300 | 0.753 | 0.723 |

No setting reached 80% training-fold accuracy while detecting both rare classes. The fallback selected the highest accuracy among settings detecting both. On the reused holdout, the selected model correctly identified zero of six agricultural and zero of three built-up cases. Its accuracy gain over the previously selected model comes from forest persistence.

The selected model reached **0.300 test macro-F1** versus 0.304 for always predicting forest persistence. Its accuracy was 81.8% versus 83.6% for the majority rule. The previous workflow already inspected this holdout. No test-driven tuning was performed. The training-only search compared 18 new settings. The previous XGBoost achieved 54.5% accuracy and 0.329 macro-F1 on this same holdout. Reuse of these parcels limits confidence in the comparison.

| outcome | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| no_change | 0.833 | 0.978 | 0.900 | 46.000 |
| to_agricultural | 0.000 | 0.000 | 0.000 | 6.000 |
| to_built_up | 0.000 | 0.000 | 0.000 | 3.000 |

Built-up recall rests on 3 test cases. One additional correct prediction changes it by 33%.

## Explanation of the full model

protected_area_0, dist_to_road_km, dist_to_city_km had the largest mean absolute SHAP contributions to the combined conversion score. Indicator suffixes on protected_area denote one-hot categories. Positive SHAP contributions increase a parcel's score relative to the training background. These associations cannot establish causation. The permutation chart measures held-out performance of the evaluation fit and may divide importance among correlated predictors.

## Retrospective parcel ranking

The five-fold ranking scores each eligible parcel while excluding it from fitting. The separately saved pipeline trained on all 275 eligible parcels scores the ten outside-scope parcels under the restricted outcome definition. Among 285 parcels, 4 are High (at least 0.66), 150 Medium (0.33 to below 0.66) and 131 Low (below 0.33).

| region | risk_score | most_likely_conversion_type | most_likely_overall_outcome | scope_status |
| --- | --- | --- | --- | --- |
| 312 | 0.780 | to_agricultural | to_agricultural | in_scope |
| 344 | 0.713 | to_built_up | to_built_up | in_scope |
| 898 | 0.688 | to_built_up | to_built_up | in_scope |
| 134 | 0.671 | to_agricultural | to_agricultural | in_scope |
| 428 | 0.641 | to_agricultural | to_agricultural | in_scope |
| 868 | 0.634 | to_agricultural | to_agricultural | in_scope |
| 903 | 0.631 | to_agricultural | to_agricultural | in_scope |
| 296 | 0.628 | to_agricultural | to_agricultural | in_scope |
| 111 | 0.616 | to_built_up | to_built_up | in_scope |
| 908 | 0.609 | to_agricultural | to_agricultural | in_scope |

## Limits and future improvements

The scores are uncalibrated screening indicators. The parcel split measures performance within one decade. Spatial and future-year performance are unknown. Built-up combines residential and industrial conversion, and these labels do not measure sustainability. With this CSV, repeated nested stratified splits could assess split sensitivity. Calibration would require separate validation, particularly for the rare outcomes.
