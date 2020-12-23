import json
import logging
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/ddosPrediction', methods=['POST'])
def ddosPrediction():
    path =''
    json_ = request.json
    print('json:', json_)
    logger.info('json_ %s' % json_)
    # print('json_g---->',json_g)
    json_obj = json.dumps(json_)
    print(json_obj)
    fwdSegSizeMin = json_['Fwd Seg Size Min']
    print("fwdSegSizeMin------------>",fwdSegSizeMin)
    initBwdWinByts = json_['Init Bwd Win Byts']
    print("initBwdWinByts----------->",initBwdWinByts)
    initFwdWinByts = json_['Init Fwd Win Byts']
    print("initFwdWinByts----------->", initFwdWinByts)
    fwdSegSizeMin = json_['Fwd Seg Size Min']
    print("fwdSegSizeMin----------->", fwdSegSizeMin)
    fwdPktLenMean = json_['Fwd Pkt Len Mean']
    print("fwdPktLenMean----------->", fwdPktLenMean)
    fwdSegSizeAvg = json_['Fwd Seg Size Avg']
    print("fwdSegSizeAvg----------->", fwdSegSizeAvg)

    ddosInput = pd.DataFrame([json_])
    test = pd.DataFrame()
    x = ddosInput[['Fwd Seg Size Avg','Fwd Seg Size Min',
                    'Init Fwd Win Byts','Init Bwd Win Byts',
                   'Fwd Seg Size Min']]


    print('query_df_array::----->', x)
    prediction = ddos_estimator.predict(x)
    print('Predicted Value--->', prediction)
    logger.info('json_ %s' % json_obj)
    # adding imagekit lib for capturing scree shot on server
    options = {'xvfb': ''}
    # inputurl = str(inputurl)
    output = prediction
    return str(output)


#MODEL_FILE = 'ddos_clf_model.pkl'
MODEL_FILE = 'ddos_clf_modelV2.pkl'
ddos_estimator = joblib.load(MODEL_FILE)
if __name__ == '__main__':
    logging.basicConfig(filename="MaliciouusUrlDetection.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_estimator = joblib.load(MODEL_FILE)
    app.run(port=8086)

