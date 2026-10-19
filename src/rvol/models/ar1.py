"""First-order autoregression on one independently prepared current value."""

from rvol.models._validation import validated_features
from rvol.models.contracts import FloatArray
from rvol.models.linear_regression import LinearPredictor, LinearRegressionModel


class AR1Model:
    """Regress the next target on one current-value input, in supplied units."""

    name = "AR1"

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        matrix = validated_features(features, n_features=1)
        return LinearRegressionModel().fit(matrix, target)
