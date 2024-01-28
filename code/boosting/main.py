import os
import argparse
import wandb
import sys

from boosting.args import parse_args
from boosting import trainer
from boosting.utils import get_logger, set_seeds, logging_conf
from boosting.dataloader import Preprocess


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    
    wandb.login()
    set_seeds(args.seed)

    logger.info("#### Data Loading ####")
    preprocess = Preprocess(args)
    preprocess.load_file(train_file_name=args.train_file_name, test_file_name=args.test_file_name)
    train_data = preprocess.get_train_data()
    test_data = preprocess.get_test_data()
    wandb.init(project=f"dkt_boosting", config=vars(args))

    logger.info(f"#### Model Loading : {args.model} ####")
    model = trainer.get_model(args, train_data)
    
    if args.kfold:
        train_data_list = preprocess.kfold_split_data(train_data)
        logger.info("#### Start Kfold Training ####")
        trainer.kfold_train(args, train_data_list, model=model)
        logger.info(f"#### Inference : {args.model} ####")
        trainer.kfold_inference(args, test_data)
    else:
        train_data = preprocess.split_data(train_data)
        logger.info("#### Start Training ####")
        trainer.train(args, train_data, model=model)
        logger.info(f"#### Inference : {args.model} ####")
        trainer.inference(args, test_data, model=model)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
