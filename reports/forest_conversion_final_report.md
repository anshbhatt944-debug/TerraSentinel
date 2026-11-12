# Forest conversion benchmark

## Data and prediction target

The only source was land_use_change.csv with 1000 parcels. The 285 forest parcels ended as forest (230), cropland (29), built-up (16), grassland (8) or barren (2). The first three classes define the model. Ten outside-scope parcels appear in the ranking only. Built-up cannot be divided into residential and industrial. Neither the outcome nor the model measures sustainability.

Soil suitability combines quality and slope. Road accessibility uses distance to roads. Population growth timing is unknown and excluded. Agricultural destination is observed but agricultural activity history and infrastructure change are unavailable. Undated baseline inputs are assumed sufficiently static. Fold-local preprocessing handles missing values and categoricals.

## Experiment and results

Seven classifiers and the majority reference used identical validation splits. Each of 84 learned configurations was evaluated across 12 training folds. Model fitting, optional PCA, and class balancing used training-fold rows only. The winning family was **Logistic Regression**, with training-selected settings `{'model__C': 1.0, 'reduce': 'passthrough', 'weight_power': 0.65}`. Sigmoid calibration selected: False. This is an exploratory search on 220 parcels. CV selection scores can be optimistic. The 55-parcel holdout was also used for reporting in earlier iterations and is not independent confirmation.

| model | CV_accuracy | CV_macro_F1 | accuracy | balanced_accuracy | macro_precision | macro_F1 | weighted_F1 | macro_OVR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.720 | 0.396 | 0.709 | 0.435 | 0.401 | 0.412 | 0.726 | 0.744 |
| Decision Tree | 0.815 | 0.322 | 0.800 | 0.423 | 0.364 | 0.391 | 0.757 | 0.618 |
| Random Forest | 0.782 | 0.326 | 0.818 | 0.430 | 0.454 | 0.436 | 0.781 | 0.708 |
| Extra Trees | 0.661 | 0.328 | 0.673 | 0.565 | 0.455 | 0.470 | 0.718 | 0.783 |
| XGBoost | 0.782 | 0.357 | 0.818 | 0.430 | 0.393 | 0.410 | 0.769 | 0.734 |
| Gradient Boosting | 0.795 | 0.347 | 0.836 | 0.486 | 0.732 | 0.509 | 0.808 | 0.734 |
| HistGradientBoosting | 0.692 | 0.378 | 0.655 | 0.365 | 0.344 | 0.348 | 0.679 | 0.686 |
| Majority baseline | 0.836 | 0.304 | 0.836 | 0.333 | 0.279 | 0.304 | 0.762 | 0.500 |

The winner correctly classified 39 of 55 parcels. The majority rule accuracy was 83.6%. The earlier project XGBoost had 54.5% accuracy and 0.329 macro-F1 on the same split. The earlier accuracy-focused trial reached 81.8% with zero correct agricultural or built-up classifications.

| outcome | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| no_change | 0.860 | 0.804 | 0.831 | 46.000 |
| to_agricultural | 0.143 | 0.167 | 0.154 | 6.000 |
| to_built_up | 0.200 | 0.333 | 0.250 | 3.000 |

Conditional stratified bootstrap 95% intervals: accuracy 0.600 to 0.800, macro-F1 0.265 to 0.592. Only three test parcels became built-up. Intervals omit training and selection uncertainty. Model timings and the training-to-test performance gaps are in the benchmark CSV.

## Model interpretation and screening

The highest mean absolute conversion SHAP contributions were dist_to_road_km, soil_suitability, protected_area. The fitted pipelines for every family appear in forest_conversion_results/fitted_families. Their full-data fits are for inference only. The benchmark table reports the evaluation fits. Separate plots explain agricultural and built-up scores. They describe model behaviour. They do not identify causes.

There are 39 High, 138 Medium and 108 Low scores. Fixed thresholds are 0.66 and 0.33. Each eligible parcel was scored outside its fitting fold. The ten outside-scope parcels use the full-data fit. Probabilities are conditional on the three-class task and retrospective; generalisation to future years is untested.

| region | risk_score | observed_2025_land_cover | most_likely_conversion_type |
| --- | --- | --- | --- |
| 910 | 0.922 | Forest | to_built_up |
| 800 | 0.903 | Built-up | to_built_up |
| 898 | 0.876 | Forest | to_built_up |
| 312 | 0.868 | Forest | to_built_up |
| 134 | 0.855 | Forest | to_agricultural |
| 467 | 0.850 | Forest | to_built_up |
| 111 | 0.834 | Forest | to_built_up |
| 336 | 0.824 | Forest | to_built_up |
| 249 | 0.815 | Built-up | to_agricultural |
| 369 | 0.813 | Cropland | to_built_up |

The model bundle needs forest_pipeline.py on the Python path. Use monitor.py with incoming baseline batches to compare feature distributions and, when mature observed outcomes are supplied, evaluate class performance. New independent outcomes are needed for prospective validation.
