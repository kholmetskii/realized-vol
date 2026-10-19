"""Heterogeneous autoregression on independently prepared numerical summaries."""

from rvol.models._validation import validated_features
from rvol.models.contracts import FloatArray
from rvol.models.linear_regression import LinearPredictor, LinearRegressionModel


class HARModel:
    """Regress targets on current, short-mean and long-mean inputs, in that order.

    The caller prepares the summaries and chooses their windows and units.
    """

    name = "HAR"

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        matrix = validated_features(features, n_features=3)
        return LinearRegressionModel().fit(matrix, target)
