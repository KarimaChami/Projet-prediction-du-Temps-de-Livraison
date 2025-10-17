import pytest
import pandas as pd
from delivery_model import run_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

@pytest.fixture 
def dataset():
    df = pd.read_csv("data/dataset.csv")
    return df

def test_dimension_split(dataset):
    X = dataset.drop(columns='Delivery_Time_min')
    y = dataset['Delivery_Time_min']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == len(y_train), "X_train et y_train n'ont pas le même nombre de lignes"
    assert len(X_test) == len(y_test), "X_test et y_test n'ont pas le même nombre de lignes"
    
    print("Dimensions cohérentes après split")

def test_MAE(dataset):
   X = dataset.drop(columns='Delivery_Time_min')
   y = dataset['Delivery_Time_min']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   model = RandomForestRegressor(n_estimators=100, random_state=42)
   param_grid = {'model__n_estimators': [100]} 
   mae, r2 = run_pipeline(model, param_grid, "Test_RF", return_scores=True)
   assert mae < 5, f" MAE trop élevée ({mae:.2f} min)"
   print(f" MAE correcte ({mae:.2f} min) - Test réussi")