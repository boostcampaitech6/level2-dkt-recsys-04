import os
import wandb
import numpy as np

from .metric import get_metric
from .model import *
from .utils import get_logger, logging_conf, get_save_time


logger = get_logger(logger_conf=logging_conf)


def train(args, train_data, model):
    
    if args.model.lower() == 'lgbm':
        result = model.fit(train_data)
    else:
        result = model.fit(train_data['X_train'], train_data['y_train'])
    
    # Train AUC / ACC
    predict = result.predict(train_data['X_valid'])
    auc, acc = get_metric(train_data['y_valid'], predict)
    
    wandb.log(dict(epoch=args.n_estimators,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc))
    
    logger.info("Valid AUC : %.4f ACC : %.4f", auc, acc)
    
def kfold_train(args, train_data_list: list, model):
    
    auc_list = []
    acc_list = []
    for fold, train_data in enumerate(train_data_list):
        if args.model.lower() == 'lgbm':
            result = model.fit(train_data)
        else:
            result = model.fit(train_data['X_train'], train_data['y_train'])
        
        # Train AUC / ACC
        predict = result.predict(train_data['X_valid'])
        auc, acc = get_metric(train_data['y_valid'], predict)
        auc_list.append(auc)
        acc_list.append(acc)
        
        # wandb.log(dict(epoch=args.num_boost_round,
        #             valid_auc_epoch=auc,
        #             valid_acc_epoch=acc))
        
        # 모델 저장
        os.makedirs(name=args.model_dir, exist_ok=True)
        model.save_model(f'{args.model_dir}{args.model}_{fold + 1}.txt')
        
        logger.info(f"  Fold {fold + 1} > Valid AUC : %.4f ACC : %.4f", auc, acc)
    
    logger.info(f"  Fold Average{fold + 1} > Valid AUC : %.4f ACC : %.4f", np.mean(auc_list), np.mean(acc_list))

def inference(args, test_data, model) -> None:

    X = test_data[args.X_columns]
    predict = model.predict(X)
    
    save_time = get_save_time()
    write_path = os.path.join(args.output_dir, f"submission_{save_time}_{args.model}" + ".csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(predict):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)

def kfold_inference(args, test_data, model):
    
    predict = np.zeros((test_data.shape[0],))
    for fold in range(args.n_fold):
        
        # 모델 불러오기
        if args.model.lower() == 'lgbm':
            model = lgb.Booster(model_file=f'{args.model_dir}{args.model}_{fold + 1}.txt')
            
        current_pred = model.predict(test_data[args.X_columns])
        predict += current_pred
    predict = predict / args.n_fold
    
    save_time = get_save_time()
    write_path = os.path.join(args.output_dir, f"submission_{save_time}_kfold_{args.model}" + ".csv")
    os.makedirs(name=args.output_dir, exist_ok=True)
    
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(predict):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved Kfold submission as %s", write_path)

def get_model(args, data):
    
    try:
        model_name = args.model.lower()
        if model_name == 'adaboost':
            model = AdaBoost(args)
        if model_name == 'gradboost':
            model = GradBoost(args)
        if model_name == 'xgboost':
            model = XGBoost(args)
        if model_name == 'catboost':
            model = CatBoost(args)
        if model_name == 'lgbm':
            model = LGBM(args)
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name)
        raise e

    return model
