import pandas as pd
from sklearn.datasets import load_wine

def load_data():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df