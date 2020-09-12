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
    image = request.args.get('path', None)
    ans = detect.infer(image)
    result = json.dumps({'status': 'SUCCESS',
                         'result': ans})
    return (result, 200)


if __name__ == '__main__':
    global classifier_ML, classifier_DL
    classifier_DL = DLClassifier(config['DL_model_path'])
    classifier_ML = MLCLassifier(config['ML_model_path'])
    print('Models Loaded')
    app.run(debug=True)
