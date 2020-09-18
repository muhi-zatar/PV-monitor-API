import pandas as pd
from sklearn import preprocessing

def normalize(data, mode, model=None):
    if mode == 'train':
        x = pd.read_excel(data, 'inputs')
        scaler = preprocessing.StandardScaler().fit(x)
        return scaler
    else:
        return model.transform([data])
