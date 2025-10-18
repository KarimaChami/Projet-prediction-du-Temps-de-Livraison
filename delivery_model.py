import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression

# Chargement des données

def load_data():
    return pd.read_csv("data/dataset.csv").copy()
data=load_data()

def split_data(df, target='Delivery_Time_min'):
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Remplir les valeurs manquantes
data["Courier_Experience_yrs"] = data["Courier_Experience_yrs"].fillna(data["Courier_Experience_yrs"].mean())
for col in ['Weather', 'Traffic_Level', 'Time_of_Day']:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Supprimer les colonnes inutiles
data = data.drop(columns=['Order_ID', 'Vehicle_Type'])

# Définir les colonnes
num_col = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
cat_col = ['Weather', 'Traffic_Level', 'Time_of_Day']


# Séparation X / y 
X = data.drop(columns='Delivery_Time_min')
y = data['Delivery_Time_min']
X_train, X_test, y_train, y_test = split_data(data)

# Encodage 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)
    ]
)

# Pipeline
def run_pipeline(model, param_grid, model_name,return_scores=False):
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('select', SelectKBest(score_func=f_regression, k=5)),
        ('model', model)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    min = int(mae)
    sec = int((mae - min) * 60)

    print(f"\n**** {model_name} Results ****")
    print("Best params:", grid.best_params_)
    print(f"Best CV MAE: {-grid.best_score_:.3f}")
    print(f"Test MAE: {mae:.3f} : {min} min {sec} s d’erreur moyenne")
    print(f"Test R²: {r2:.3f}")

    if return_scores:
        return mae, r2

param_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None]
}

param_svr = {
    'model__kernel': ['linear', 'rbf'],
    'model__C': [0.1, 1, 10],
    'model__gamma': ['scale', 'auto']
}

run_pipeline(RandomForestRegressor(), param_rf, "RandomForestRegressor")
run_pipeline(SVR(), param_svr, "SVR")
 
