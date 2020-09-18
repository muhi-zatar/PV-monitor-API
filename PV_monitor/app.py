import os
import json
import numpy as np
from flask import Flask, request, Response
from ML_classifier import MLClassifier
from DL_classifier import DLClassifier
import sys
#sys.path.append('/home/mawdoo3/Muhystuff/research/PV_images/PV_thermal')
from config import config

app = Flask(__name__)

@app.route('/classifier', methods=['GET','POST'])

def classifier():
    global classifier_ML, classifier_DL
    irradiance = request.args.get('irradiance', None)
    temperature = request.args.get('voltage', None)
    voltage = request.args.get('voltage', None)
    current = request.args.get('current', None)
    power = request.args.get('power', None)

    if config['model'] == 'ML':
        ans = classifier_ML.infer(irradiance,
                                  temperature,
                                  voltage,
                                  current,
                                  power)

    elif config['model'] == 'DL':
        ans = classifier_DL.infer(irradiance,
                                  temperature,
                                  voltage,
                                  current,
                                  power)
    else:
        raise ValueError('Undefined Model/ Inferencing type for {}'.format(model))

    result = json.dumps({'status': 'SUCCESS',
                         'result': ans})

    return (result, 200)


if __name__ == '__main__':
    global classifier_ML, classifier_DL
    if config['model'] == 'DL':
        classifier_DL = DLClassifier(config['DL_model_path'])
    else:
        classifier_ML = MLCLassifier(config['ML_model_path'])
    print('Model Loaded')
    app.run(debug=True)
