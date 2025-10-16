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


from sklearn.feature_selection import SelectKBest, f_classif, f_regression
def selectKBest(X_train,X_test,y_train,k=5):
   selector = SelectKBest(score_func=f_regression, k=k)
   X_train_selected = selector.fit_transform(X_train, y_train)
   X_test_selected = selector.transform(X_test)
   selected_cols = X_train.columns[selector.get_support()]
   return X_train_selected, X_test_selected, selected_cols

X_train_selected, X_test_selected, selected_cols = selectKBest(X_train, X_test, y_train, k=5)
print("best Features sélectionnees :", selected_cols.tolist())

def train_models(X_train,y_train):
    models = {
        'Logistic Regression': RandomForestRegressor(),
        'Random Forest': SVR()
    }
    for name in models:
        models[name].fit(X_train, y_train)
    return models

from sklearn.feature_selection import f_regression
 

