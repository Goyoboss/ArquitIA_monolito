from datapath import load_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = load_data()

def entrenamiento(df, objetivo, model):
    # Matriz de características y vector objetivo
    X = df.drop(objetivo, axis=1)
    y = df[objetivo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    pre_train = precision_score(y_train, y_pred_train, average="weighted")
    pre_test = precision_score(y_test, y_pred_test, average="weighted")

    rec_train = recall_score(y_train, y_pred_train, average="weighted")
    rec_test = recall_score(y_test, y_pred_test, average="weighted")

    f1_train = f1_score(y_train, y_pred_train, average="weighted")
    f1_test = f1_score(y_test, y_pred_test, average="weighted")

    cm = confusion_matrix(y_test, y_pred_test)

    print("-"*10, f"METRICAS EN {model}", "-"*10)
    print(f"Accuracy en entrenamiento: {acc_train:.2f}\nAccuracy en prueba: {acc_test:.2f}\n")
    print(f"Precision en entrenamiento: {pre_train:.2f}\nPrecision en prueba: {pre_test:.2f}\n")
    print(f"Recall en entrenamiento: {rec_train:.2f}\nRecall en prueba: {rec_test:.2f}\n")
    print(f"F1 en entrenamiento: {f1_train:.2f}\nF1 en prueba: {f1_test:.2f}\n")
    print("Matriz de confusión:\n", cm)

Regresion_Logistica = LogisticRegression(max_iter=1000) # 'l2' ya es el valor por defecto
Arbol_Decision = DecisionTreeClassifier(max_depth=10, min_samples_split=30, min_samples_leaf=7)

reglog = entrenamiento(df, "target", Regresion_Logistica)
print(reglog)
arbol = entrenamiento(df, "target", Arbol_Decision)
print(arbol)