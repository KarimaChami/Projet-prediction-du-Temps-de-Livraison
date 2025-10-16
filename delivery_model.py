# #cleaning + model + GridSearchCV
# def trouver_manquant():
#     # liste = []
#     nums = input("entrer les nombres separer par des vergules :")
#     liste = [int(x) for x in nums.split(",")]
#     liste.sort()
#     for i in range(len(liste) - 1):
#         if liste[i+1] != liste[i] + 1:
#             print(f"Le nombre manquant est : {liste[i] + 1}")
# trouver_manquant()

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
# from sklearn. import RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
#Fonctions de préparation des données'
def load_data(path):
  dt =  pd.read_csv(f"{path}.csv")
  return dt
# print(data)
def clean_data(data):
 data = data.copy()
 data["Courier_Experience_yrs"] = data["Courier_Experience_yrs"].fillna(data["Courier_Experience_yrs"].mean())
 cat_cols = ['Weather','Traffic_Level','Time_of_Day']
 for col in cat_cols :
      data[col].fillna(data[col].mode()[0],inplace=True)
 return data
data = load_data("data/dataset")
data = clean_data(data)
# print("Valeurs manquantes restantes :")
# print(data.isnull().sum())



def Encodage(data):
 data = data.drop(columns=['Order_ID','Vehicle_Type'])
 
 num_col = data.select_dtypes(include=["int64","float64"]).columns
 cat_col = data.select_dtypes(include=["object"]).columns
 
 cat_encoder = OneHotEncoder(sparse_output=False)
 num_transformer = StandardScaler()
 
 for col in num_col :
   data[col] = num_transformer.fit_transform(data[[col]])
 
 for col in cat_col :
   encoded = cat_encoder.fit_transform(data[[col]])
   encoded_df = pd.DataFrame(encoded, columns=cat_encoder.get_feature_names_out([col]))
   data = data.drop(columns=[col]).join(encoded_df)

 return data 
data = Encodage(data)
# print(data)


def split_data(data, target='Delivery_Time_min'):
    X = data.drop(columns=[target])
    y = data[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
# split_data()
X_train, X_test, y_train, y_test = split_data(data)

num_col = X_train.select_dtypes(include=["float64"]).columns
scaler = StandardScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])

# def Normalisation(X_train,X_test,num_col):
#    scaler = StandardScaler()
#    X_train_scaled = X_train.copy()  # Copier pour ne pas modifier l'original
#    X_test_scaled = X_test.copy()
#    X_train_scaled[num_col] = scaler.fit_transform(X_train[num_col]) # Normaliser uniquement les colonnes numériques
#    X_test_scaled[num_col] = scaler.transform(X_test[num_col])
#    return X_train_scaled,X_test_scaled
# X_train_scaled,X_test_scaled = Normalisation(X_train,X_test,num_col)


from sklearn.feature_selection import SelectKBest, f_regression
def selectKBest(X_train,X_test,y_train,k=5):
   selector = SelectKBest(score_func=f_regression, k=k)
   X_train_selected = selector.fit_transform(X_train, y_train)
   X_test_selected = selector.transform(X_test)
   selected_cols = X_train.columns[selector.get_support()]
   return X_train_selected, X_test_selected, selected_cols

X_train_selected, X_test_selected, selected_cols = selectKBest(X_train, X_test, y_train, k=5)
print("best Features sélectionnees :", selected_cols.tolist())

# def train_models(X_train,y_train):
#     models = {
#         'Logistic Regression': RandomForestRegressor(),
#         'Random Forest': SVR()
#     }
#     for name in models:
#         models[name].fit(X_train, y_train)
#     return models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Modèle 1 : RandomForestRegressor

rfr = RandomForestRegressor(random_state=42)
rfr_param = {
   'n_estimators':[100,200],   #The number of trees in the forest.
   'max_depth':[5,10,None]  #The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
}

rfr_grid = GridSearchCV(rfr,param_grid=rfr_param,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,verbose=1)
rfr_grid.fit(X_train_selected,y_train)

## evaluation
from sklearn.metrics import mean_absolute_error, r2_score 
y_pred_rfr = rfr_grid.predict(X_test_selected)
rfr_mae = mean_absolute_error(y_test,y_pred_rfr)
rfr_r2 = r2_score(y_test,y_pred_rfr)

print("RandomForestRegressor")
print("Best params :", rfr_grid.best_params_)
print("Best CV score (MAE):", -rfr_grid.best_score_)
print("Test MAE:", rfr_mae)
print("Test R²:", rfr_r2)
print("-" * 40)

# Modèle 2 : SVR

svr = SVR()
svr_param = {
   'kernel': ['linear', 'rbf'],  #Définit le type de fonction utilisée pour transformer les données afin de trouver une frontière optimale.
    'C': [0.1, 1, 10], #Contrôle le niveau de pénalisation des erreurs.
    'gamma': ['scale', 'auto'] #Contrôle l’influence d’un seul point de donnée sur la forme du modèle.
}
svr_grid = GridSearchCV(svr,param_grid=svr_param,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,verbose=1)
svr_grid.fit(X_train_selected,y_train)
## evaluation
y_pred_svr = svr_grid.predict(X_test_selected)
svr_mae = mean_absolute_error(y_test,y_pred_svr)
svr_r2 = r2_score(y_test,y_pred_svr)

print("SVR")
print("Best params :", svr_grid.best_params_)
print("Best CV score (MAE):", -svr_grid.best_score_)  #MAE moyen pendant la validation croisée (découpé en 5 partie) C’est une estimation interne de la performance du modèle, obtenue pendant la recherche des meilleurs hyperparamètres.
print("Test MAE:", svr_mae)#C’est la vraie erreur du modèle final sur de nouvelles données (performance réelle en production)./Tu fais une vraie prédiction sur des données jamais vues (X_test).
print("Test R²:", svr_r2)
print("-" * 40)

'''
Les scores CV et Test sont proches, donc pas de surapprentissage (overfitting).
Exemple :
RF : 0.369 → 0.337 → différence faible 
SVR : 0.326 → 0.288 → très cohérent aussi 
'''
cat_col = data.select_dtypes(include=["object"]).columns



preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_col),
        ('cat',OneHotEncoder(),cat_col),

    ]
)
def run_pipline(model,param_grid,model_name):
    pipeline = Pipeline(steps=[
        ('preprocessing',preprocessor),
        ('select',SelectKBest(score_func=f_regression,k=5)),
        ('model',model)
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
    r2 = r2_score(y_test,y_pred)
    print(f"\n****{model_name} Results ****")
    print("Best params:", grid.best_params_)
    print("Best CV MAE:", -grid.best_score_)
    print("Test MAE:", mae)
    print("Test R²:", r2)
# RandomForestRegressor
param_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None]
}

# SVR
param_svr = {
    'model__kernel': ['linear', 'rbf'],
    'model__C': [0.1, 1, 10],
    'model__gamma': ['scale', 'auto']
}

run_pipline(RandomForestRegressor(),param_rf,"RandomForestRegressor")
run_pipline(SVR(),param_svr,"SVR")