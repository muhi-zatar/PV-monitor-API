import os
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

from keras.models import load_model
from utils import normalize
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class DLClassifier(object):
    def __init__(self, model_path):
        self.session = tf.Session()
        keras.backend.set_session(self.session)
#        self.data = config['data']
        self.data = 'data/Data.xlsx'
        self.normalization = normalize(data=self.data,
                                       mode='train',
                                       model=None)
        self.model = tf.keras.models.load_model(model_path)
        self.model._make_predict_function()
        self.outputs = {0 : 'Normal Operation',
                        1 : 'Panel Degradation',
                        2 : 'Partial Shading'}

    def infer(self, irradiance, temperature, voltage, current, power):
        with self.session.as_default():
            with self.session.graph.as_default():
                inputs = np.array([irradiance, temperature, voltage, current, power])
                electrical = normalize(data=inputs,
                                       mode='test',
                                       model=self.normalization)
                pred = self.model.predict(electrical)
                ans = np.argmax(pred, axis=1)
        return self.outputs[int(ans)]
