from datapath import df
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def entrenamiento(df, objetivo, model):
    #Matriz de características
    X = df.drop(objetivo, axis=1)
    #Vector objetivo
    y = df[objetivo]

    scaler = StandardScaler()

    X_processed = scaler.fit_transform(X)

    #División del conjunto de datos en 80-20
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # Agregamos pos_label='Yes' para que reconozca la clase positiva correcta
    pre_train = precision_score(y_train, y_pred_train, pos_label='Yes')
    pre_test = precision_score(y_test, y_pred_test, pos_label='Yes')

    rec_train = recall_score(y_train, y_pred_train, pos_label='Yes')
    rec_test = recall_score(y_test, y_pred_test, pos_label='Yes')

    f1_train = f1_score(y_train, y_pred_train, pos_label='Yes')
    f1_test = f1_score(y_test, y_pred_test, pos_label='Yes')

    cm = confusion_matrix(y_test, y_pred_test)

    print("-"*10, f"METRICAS EN {model}", "-"*10)

    print(f"Accuracy en entrenamiento: {acc_train:.2f}\nAccuracy en prueba: {acc_test:.2f}\n")
    print(f"Precision en entrenamiento: {pre_train:.2f}\nPrecision en prueba: {pre_test:.2f}\n")
    print(f"Recall en entrenamiento: {rec_train:.2f}\nRecall en prueba: {rec_test:.2f}\n")
    print(f"F1 en entrenamiento: {f1_train:.2f}\nF1 en prueba: {f1_test:.2f}\n")
    print(f"Matriz de confusión\n", cm)

Regresion_Logistica = LogisticRegression(max_iter=1000, penalty='l2')
Arbol_Decision = DecisionTreeClassifier(max_depth=10, min_samples_split=30, min_samples_leaf=7)

entrenamiento(df, "Churn", Regresion_Logistica)

entrenamiento(df, "Churn", Arbol_Decision)