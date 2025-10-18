import pytest
import pandas as pd
from delivery_model import run_pipeline,load_data
from sklearn.svm import SVR


@pytest.fixture 
def dataset():
    return load_data()

def test_dimension_split(dataset):
    assert isinstance(dataset, pd.DataFrame), "La variable 'data' doit être un DataFrame"
    assert dataset.shape[0] > 0, "Le DataFrame est vide (0 lignes)"
    assert dataset.shape[1] > 1, "Le DataFrame doit avoir plusieurs colonnes"

def test_pipeline_MAE():
   model = SVR(kernel='linear')
   param_grid = {'model__kernel':['linear']} 
   mae, r2 = run_pipeline(model, param_grid, "Test_SVR", return_scores=True)
   assert mae < 7, f" MAE trop élevée ({mae:.2f} min)"
   print(f" MAE correcte ({mae:.2f} min) - Test passed")