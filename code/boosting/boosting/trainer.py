import os
import wandb

from .metric import get_metric
from .model import *
from .utils import get_logger, logging_conf, get_save_time


logger = get_logger(logger_conf=logging_conf)


def train(args, data, model):
    
    result = model.fit(data['X_train'], data['y_train'])
    predict = result.predict(data['X_valid'])
    
    # Train AUC / ACC
    auc, acc = get_metric(data['y_valid'], predict)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)

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
            model = LGBM(args, data)
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name)
        raise e

    return model
