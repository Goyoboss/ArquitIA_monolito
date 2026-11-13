#El mejor modelo fue regresión logística, pero no entiendo porque sale el 100%

from sklearn.metrics import confusion_matrix
from Entrenamiento import entrenamiento
from datapath import df
from sklearn.linear_model import LogisticRegression

print("El mejor modelo fue el clasificador con Regresión logística")

Regresion_Logistica = LogisticRegression(max_iter=1000, penalty='l2')
entrenamiento(df, "Churn", Regresion_Logistica)