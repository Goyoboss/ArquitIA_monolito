#El mejor modelo fue regresión logística, pero no entiendo porque sale el 100%
from Entrenamiento import reglog
from datapath import load_data

df = load_data()

print("El mejor modelo fue el clasificador con Regresión logística")

print(reglog)