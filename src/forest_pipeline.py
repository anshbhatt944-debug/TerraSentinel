"""
TerraSentinel Pipeline with training-fold-only inverse frequency weights.
"""
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

class ForestPipeline(Pipeline):
    def __init__(self, steps, weight_power=0.5):
        super().__init__(steps)
        self.weight_power = weight_power

    def fit(self, X, y, **params):
        if self.weight_power:
            weights = compute_sample_weight("balanced", y) ** self.weight_power
            params["model__sample_weight"] = weights / weights.mean()
        return super().fit(X, y, **params)

