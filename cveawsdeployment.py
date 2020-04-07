from sklearn.externals import joblib
import pandas as pd
from sklearn import tree, model_selection, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import os
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    #print('json---------------->>>>>>>>>>>>>>:')
    json_ = request.json
    print('json:', json_)
    inputData = json_
    url_test = pd.DataFrame([json_])
    print('url_test item data---->',url_test['data'])
    df = pd.read_csv(
        "CveScore.csv",
        encoding='ISO-8859-1')

    df.dropna(subset=["baseScore"], inplace=True)

#    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
#                            stop_words='english')
#    features = tfidf.fit_transform(df.description).toarray()
#    #labels = df.baseScore
#    features.shape
    X_train, X_test, y_train, y_test = train_test_split(df['description'], df['baseScore'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
   # print("before logestimator------------------>",count_vect.transform(url_test['data']))
    #print(log_estimator.predict(count_vect.transform(url_test['data'])))
    output = log_estimator.predict(count_vect.transform(url_test['data']))
    return jsonify(pd.Series(output).to_json(orient='values'))
   

MODEL_FILE = 'regression_model-v4.pkl'
log_estimator = joblib.load(MODEL_FILE)
if __name__ == '__main__':
    log_estimator = joblib.load(MODEL_FILE)
    app.run(port=8086)

