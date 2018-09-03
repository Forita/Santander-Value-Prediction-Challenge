import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")
import lightgbm as lgb
from bayes_opt import BayesianOptimization


train_X = train.drop(["ID", "target"], axis=1)
test_X = test.drop(["ID"], axis=1)
train_y = np.log1p(train["target"].values)

def lgbm_evaluate(**params):
    warnings.simplefilter('ignore')
    
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
        
    clf = lgb.LGBMClassifier(**params, n_estimators = 10000, nthread = 4)

    folds = KFold(n_splits = 5, shuffle = True, random_state = 1001)
        
    test_pred_proba = np.zeros(train_X.shape[0])
    
    feats = [f for f in train.columns if f not in ['Target','ID']]
    
    for dev_index, val_index in kf.split(train_X):
        train_X, valid_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
        train_y, valis_y = train_y[dev_index], train_y[val_index]

        clf.fit(train_X, train_y, 
                eval_set = [(train_X, train_y), (valid_X, valid_y)], eval_metric = 'rmsle', 
                verbose = False, early_stopping_rounds = 100)

        test_pred_proba[valid_idx] = clf.predict_proba(valid_X, num_iteration = clf.best_iteration_)[:, 1]
        
        del train_X, train_y, valid_X, valid_y
        gc.collect()

    return np.expm1(test_pred_proba)
    
  params = {'colsample_bytree': (0.8, 1),
          'learning_rate': (.01, .02), 
          'num_leaves': (33, 35), 
          'subsample': (0.8, 1), 
          'max_depth': (7, 9), 
          'reg_alpha': (.03, .05), 
          'reg_lambda': (.06, .08), 
          'min_split_gain': (.01, .03),
          'min_child_weight': (38, 40)}
bo = BayesianOptimization(lgbm_evaluate, params)
bo.maximize(init_points = 5, n_iter = 5)

best_params = bo.res['max']['max_params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])

best_params
