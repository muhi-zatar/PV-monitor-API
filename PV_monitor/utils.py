import pandas as pd
from sklearn import preprocessing

def normailze(data, mode, model=None):
    if mode == 'train':
        scaler = preprocessing.StandardScaler().fit(data)
        return scaler
    else:
        return model.transform(data)
