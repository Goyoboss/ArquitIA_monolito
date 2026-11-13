#El mejor modelo tomando en cuenta la métrica F1 (por la existencia de clases desbalanceadas) 
#es regresión logística –aunque los dos cuentan con un sobreajuste significativo

from sklearn.metrics import confusion_matrix
from Entrenamiento import entrenamiento
from datapath import df
from sklearn.linear_model import LogisticRegression

print("El mejor modelo fue el clasificador con Regresión logística")

Regresion_Logistica = LogisticRegression(max_iter=1000, penalty='l2')
entrenamiento(df, "Churn", Regresion_Logistica)