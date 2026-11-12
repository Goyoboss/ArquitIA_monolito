from datapath import df
import matplotlib.pyplot as plt
import seaborn as sns

print("Información general del dataset: \n", df.info())

#función para conocer la distribución de ciertas variables con un gráfico de pastel
def pie_plot(df, column):
  pie = plt.figure(figsize=(5,8))
  plt.pie(df[column].value_counts(), labels=df[column].value_counts().index, autopct='%1.1f%%')
  plt.title = f"Distribución de {column}"
  labels = df[column].value_counts().index
  plt.legend(labels)
  return plt.show()

# revisar algunas variables con clases desbalanceadas + distribución con un gráfico de pastel

print("\nRevisamos clases desbalanceadas con distintas variables: \n", "\n")

print(df["gender"].value_counts(), "\n",
      pie_plot(df, "gender"), "\n\n" )

print(df["SeniorCitizen"].value_counts(), "\n",
      pie_plot(df, "SeniorCitizen"), "\n\n" )

# variable objetivo
print(df["Churn"].value_counts(), "\n",
      pie_plot(df, "Churn"), "\n\n" )

def hist_plot(df, column):
  plot = sns.histplot(df[column])
  plt.title = f"Histograma"
  plt.show()
  datos = (f"Promedio: {df[column].mean():.2f}",
  f"Mediana: {df[column].median():.2f}",
  f"Moda: {df[column].mode()[0]:.2f}",
  f"Desviación estándar: {df[column].std():.2f}")
  return datos

hist_plot(df, "MonthlyCharges")

hist_plot(df, "tenure")