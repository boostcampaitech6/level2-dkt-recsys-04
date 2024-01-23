from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

import numpy as np

class AdaBoost():
    def __init__(self, args):
        self.args = args
        self.model = AdaBoostClassifier(
                                        estimator=None,
                                        n_estimators=self.args.n_estimators,
                                        learning_rate=self.args.lr_AdaBoost,
                                        algorithm='SAMME.R',
                                        random_state=args.data_shuffle,
                                        base_estimator='deprecated'
                                       )
        
    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_valid):
        return self.model.predict(X_valid)
    
class GradBoost():
    def __init__(self, args):
        self.args = args
        self.model = GradientBoostingClassifier(
                                                loss='log_loss',
                                                learning_rate=self.args.lr,
                                                n_estimators=self.args.n_estimators,
                                                subsample=self.args.subsample,
                                                criterion='friedman_mse',
                                                min_samples_split=self.args.min_samples_split,
                                                min_samples_leaf=self.args.min_samples_leaf,
                                                min_weight_fraction_leaf=self.args.min_weight_fraction_leaf,
                                                max_depth=self.args.max_depth,
                                                min_impurity_decrease=self.args.min_impurity_decrease,
                                                init=None,
                                                random_state=args.data_shuffle,
                                                max_features=None,
                                                verbose=self.args.verbose,
                                                max_leaf_nodes=None,
                                                warm_start=False,
                                                validation_fraction=self.args.validation_fraction,
                                                n_iter_no_change=None,
                                                tol=self.args.tol,
                                                ccp_alpha=self.args.ccp_alpha
                                               )
        
    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_valid):
        return self.model.predict(X_valid)
    
class XGBoost():
    def __init__(self, args):
        self.args = args
        self.model = XGBClassifier(
                                   n_estimators=self.args.n_estimators,
                                   random_state=np.random.seed(self.args.seed),
                                   max_depth=self.args.max_depth,
                                   colsample_bylevel=self.args.colsample_bylevel,
                                   colsample_bytree=self.args.colsample_bytree,
                                   gamma=self.args.gamma,
                                   min_child_weight=self.args.min_child_weight,
                                   nthread=self.args.nthread,
                                  )
        
    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_valid):
        return self.model.predict(X_valid)
    
class CatBoost():
    def __init__(self, args):
        self.args = args
        self.model = CatBoostClassifier(
                                        n_estimators=self.args.n_estimators,
                                        learning_rate=self.args.lr,
                                        random_state=np.random.seed(self.args.seed)
                                       )
        
    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_valid):
        return self.model.predict(X_valid)
    
class LGBM():
    def __init__(self, args):
        self.args = args
        self.parameter = {
                          'max_depth': self.args.max_depth,
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
    
    def predict(self, X_valid):
        return self.model.predict(X_valid)
    
    def save_model(self, file_name):
        self.model.save_model(file_name)