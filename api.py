from fastapi import FastAPI
from src.knn import KNN
from sklearn import datasets
import pandas as pd
import numpy as np

app = FastAPI()
#entrenar el modelo
iris = datasets.load_iris() #datos
X_c = iris.data 
y_c = iris.target
category = {0:'setosa', 1:'versicolor', 2: 'virginica'} #categorias
print("hola")

train_ratio = 0.75

# Calculate the number of samples for the training set
num_samples = len(X_c)
num_train_samples = int(train_ratio * num_samples)

# Create random indices for splitting
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Split the data and labels into training and testing sets
X_train_c = X_c[indices[:num_train_samples]]
y_train_c = y_c[indices[:num_train_samples]]
X_test_c = X_c[indices[num_train_samples:]]
y_test_c = y_c[indices[num_train_samples:]]

#creacion del modelo
knn=KNN(k=5,metric='euclidean') #cinco vecinos, distancia euclediana
knn.fit(X_train_c, y_train_c)

@app.get("/") #decoradores
async def root():
    return {"message": "Hola Mundo"}


@app.get("/models/{model_id}") #{data1}/{data2} rompe con el esquema del api REST
async def get_model(model_id: str, data1:float, data2:float): #son parametros de query
    return {"model_id": model_id, "data1":data1*2, "data2":data2*3}

@app.get("/models/{category}/{model_id}") 
async def get_model_category(model_id: str,category: str, data1:float = 0, data2:float=0): #son parametros default
    return {"model_id": model_id, 
            "category": category,
            "data1":data1*2, 
            "data2":data2*3}


@app.get("/predict")
async def predict(sepalLength: float, sepalWidth: float, petalLength: float, petalWidth: float):
    #ulitizar el modelo para predecir
    dato =[[sepalLength,sepalWidth,petalLength,petalWidth]]
    result = knn.predict(dato)
    classification = category.get(result[0])
    return{"prediccion":classification}




