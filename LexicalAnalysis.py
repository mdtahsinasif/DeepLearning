# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:27:12 2019

@author: tahsin.asif
"""



import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler


shift = 200


def encode(charcode):
    r = 0
    ln = len(charcode)
    if(ln > 2):
        print("Error: Expected Maximum of Two Characters!")
        exit(0)
    for i in range(ln):
        r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)
    return r


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con = 2
    x = preds-labels
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess


def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)


def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    # compute skew and do Box-Cox transformation (Tilli)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


if __name__ == "__main__":
    print('Started')
    directory = 'C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/LexicalMLExample/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')
    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    cats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)
    for col in cats:
        train_test[col] = train_test[col].apply(encode)
    train_test.loss = np.log(train_test.loss + shift)
    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)
    train = train_test.iloc[:ntrain, :].copy()
    test = train_test.iloc[ntrain:, :].copy()
    test.drop('loss', inplace=True, axis=1)

    print('Median Loss:', train.loss.median())
    print('Mean Loss:', train.loss.mean())
    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.03,
        'objective': 'reg:linear',
        'max_depth': 12,
        'min_child_weight': 100,
        'booster': 'gbtree',
    }

    # dtrain = xgb.DMatrix(train[train.columns[1:-1]].values,
    #                      label=train.loss)
    # res = xgb.cv(xgb_params, dtrain, num_boost_round=2500, nfold=10,
    #              seed=1, stratified=False,
    #              early_stopping_rounds=25,
    #              obj=logregobj,
    #              feval=xg_eval_mae, maximize=False,
    #              verbose_eval=50, show_stdv=True)

    # best_nrounds = res.shape[0] - 1
    # cv_mean = res.iloc[-1, 0]
    # cv_std = res.iloc[-1, 1]

    # print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
    # print('Best Round: {0}'.format(best_nrounds))
    # del dtrain
    # del res
    # gc.collect()
    # exit(0)

    best_nrounds = 350  # 640 score from above commented out code (Faron)
    allpredictions = pd.DataFrame()
    kfolds = 1  # 10 folds is better!
    if kfolds > 1:
        kf = KFold(train.shape[0], n_folds=kfolds)
        for i, (train_index, test_index) in enumerate(kf):
            dtest = xgb.DMatrix(test[test.columns[1:]])
            print('Fold {0}'.format(i + 1))
            X_train, X_val = train.iloc[train_index], train.iloc[test_index]
            dtrain = \
                xgb.DMatrix(X_train[X_train.columns[1:-1]],
                            label=X_train.loss)
            dvalid = \
                xgb.DMatrix(X_val[X_val.columns[1:-1]],
                            label=X_val.loss)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

            gbdt = xgb.train(xgb_params, dtrain, best_nrounds, watchlist,
                             obj=logregobj,
                             feval=xg_eval_mae, maximize=False,
                             verbose_eval=50,
                             early_stopping_rounds=25)
            del dtrain
            del dvalid
            gc.collect()
            allpredictions['p'+str(i)] = \
                gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
            del dtest
            del gbdt
            gc.collect()
    else:
        dtest = xgb.DMatrix(test[test.columns[1:]].values)
        dtrain = \
            xgb.DMatrix(train[train.columns[1:-1]].values,
                        label=train.loss)
        watchlist = [(dtrain, 'train'), (dtrain, 'eval')]
        gbdt = xgb.train(xgb_params, dtrain, best_nrounds, watchlist,
                         obj=logregobj,
                         feval=xg_eval_mae, maximize=False,
                         verbose_eval=50, early_stopping_rounds=25)
        allpredictions['p1'] = \
            gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
        del dtrain
        del dtest
        del gbdt
        gc.collect()

    print(allpredictions.head())

    submission = pd.read_csv(directory + 'sample_submission.csv')
    if(kfolds > 1):
        submission.iloc[:, 1] = \
            np.exp(allpredictions.mean(axis=1).values)-shift
        submission.to_csv('xgbmeansubmission.csv', index=None)
        submission.iloc[:, 1] = \
            np.exp(allpredictions.median(axis=1).values)-shift
        submission.to_csv('xgbmediansubmission.csv', index=None)
    else:
        submission.iloc[:, 1] = np.exp(allpredictions.p1.values)-shift
        submission.to_csv('xgbsubmission.csv', index=None)
    print('Finished')
