import pandas as pd

data = pd.read_csv("Telco_Customer_Churn.csv")
df = data.drop(columns=["customerID"])

print(df.head())