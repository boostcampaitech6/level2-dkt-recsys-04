from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

import numpy as np

class XGBoost():
    def __init__(self, args):
        self.args = args
        self.model = XGBClassifier(
                                   n_estimators=self.args.n_estimators,
                                   random_state=np.random.seed(self.args.seed),
                                   max_depth=self.args.max_depth_xgb,
                                   colsample_bylevel=self.args.colsample_bylevel,
                                   colsample_bytree=self.args.colsample_bytree,
                                   gamma=self.args.gamma,
                                   min_child_weight=self.args.min_child_weight,
                                   nthread=self.args.nthread,
                                   eval_metric='auc',
                                  )
        
    def fit(self, train_data):
        return self.model.fit(
                              train_data['X_train'], train_data['y_train'],
                              eval_set=[(train_data['X_train'], train_data['y_train']),(train_data['X_valid'], train_data['y_valid'])],
                             )
    
    def predict_proba(self, X_valid):
        return self.model.predict_proba(X_valid)[:, 1]
    
class CatBoost():
    def __init__(self, args):
        self.args = args
        self.model = CatBoostClassifier(
                                        n_estimators=self.args.n_estimators,
                                        learning_rate=self.args.lr,
                                        random_state=np.random.seed(self.args.seed),
                                        eval_metric='AUC'
                                       )
        
    def fit(self, train_data):
        return self.model.fit(train_data['X_train'], train_data['y_train'],
                              eval_set=(train_data['X_train'], train_data['y_train']),
                              early_stopping_rounds=10,
                              verbose=50,)
    
    def predict_proba(self, X_valid):
        return self.model.predict_proba(X_valid)[:, 1]
    
class LGBM():
    def __init__(self, args):
        self.args = args
        self.parameter = {
                          'max_depth': self.args.max_depth_lgbm,
                          'min_data_in_leaf': self.args.min_data_in_leaf,
                          'feature_fraction': self.args.feature_fraction,
                          'lambda': self.args._lambda,
                          'learning_rate': self.args.lr,
                          'boosting_type': 'gbdt',
                          'objective': 'binary',
                          'metric': ['auc', 'binary_logloss'],
                          'force_row_wise': True,
                          'verbose': 1,
                         }
        
    def fit(self, train_data):
        self.lgb_train = lgb.Dataset(train_data['X_train'], train_data['y_train'])
        self.lgb_valid = lgb.Dataset(train_data['X_valid'], train_data['y_valid'])
        self.model = lgb.train(
                               self.parameter,
                               self.lgb_train,
                               valid_sets=[self.lgb_train, self.lgb_valid],
                               num_boost_round=self.args.n_estimators,
                              )
        return self.model
    
    def predict_proba(self, X_valid):
        return self.model.predict(X_valid)