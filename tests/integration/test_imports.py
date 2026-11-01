import subprocess
import sys


def test_forecasting_imports_do_not_load_matplotlib():
    code = """
import sys
import pandas as pd
from rvol.data.preparation import build_next_observation_targets, prepare_forecast_data
from rvol.evaluation import forecast_losses
from rvol.models import ForecastModel, LinearRegression
from rvol.walkforward import walk_forward
data = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=20), 'value': range(20)})
features = pd.DataFrame({'origin_date': data.date, 'lag_0': data.value.astype(float)})
prepared = prepare_forecast_data(features, build_next_observation_targets(data))
model = ForecastModel('AR1', LinearRegression(), ('lag_0',))
forecasts = walk_forward(prepared, model, min_train_size=5)
forecast_losses(forecasts, lambda actual, predicted: (actual - predicted) ** 2)
assert 'matplotlib' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_importing_models_and_walk_forward_loads_no_preparation_or_evaluation():
    code = """
import sys
from rvol.walkforward import walk_forward
from rvol.models import ForecastModel, LinearRegression
for prefix in ('rvol.data', 'rvol.evaluation', 'matplotlib', 'scipy'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
