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
    preprocess.load_train_data(file_name=args.file_name)
    preprocess.load_test_data(file_name=args.test_file_name)
    train_data = preprocess.get_train_data()
    test_data = preprocess.get_test_data()
    data = preprocess.split_data(train_data)
    wandb.init(project="boosting", config=vars(args))

    logger.info(f"#### Model Loading : {args.model} ####")
    model = trainer.get_model(args, data)
    
    logger.info("#### Start Training ####")
    trainer.train(args, data, model=model)
    
    logger.info(f"#### Inference : {args.model} ####")
    trainer.inference(args, test_data, model=model)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
