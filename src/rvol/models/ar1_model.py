"""First-order autoregression on one independently prepared current value."""

from rvol.models._validation import validated_features
from rvol.models.arrays import FloatArray
from rvol.models.forecast_model import ForecastModel
from rvol.models.linear_predictor import LinearPredictor
from rvol.models.linear_regression_model import LinearRegressionModel


class AR1Model(ForecastModel):
    """Regress the next target on one current-value input, in supplied units."""

    name = "AR1"

    def fit(self, features: FloatArray, target: FloatArray) -> LinearPredictor:
        matrix = validated_features(features, n_features=1)
        return LinearRegressionModel().fit(matrix, target)
