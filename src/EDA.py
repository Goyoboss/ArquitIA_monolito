from datapath import df
import matplotlib.pyplot as plt
import seaborn as sns

print("Información general del dataset: \n", df.info())


def hist_plot(df, column):
  plot = sns.histplot(df[column])
  plt.title = f"Histograma"
  plt.show()
  datos = (f"Promedio: {df[column].mean():.2f}",
  f"Mediana: {df[column].median():.2f}",
  f"Moda: {df[column].mode()[0]:.2f}",
  f"Desviación estándar: {df[column].std():.2f}")
  return datos

hist_plot(df, "target")

hist_plot(df, "alcohol")