#El mejor modelo fue regresión logística, pero no entiendo porque sale el 100%
from Entrenamiento import entrenamiento
from datapath import load_data
from sklearn.linear_model import LogisticRegression

df = load_data()

print("El mejor modelo fue el clasificador con Regresión logística")

Regresion_Logistica = LogisticRegression(max_iter=1000, penalty='l2')
entrenamiento(df, "target", Regresion_Logistica)