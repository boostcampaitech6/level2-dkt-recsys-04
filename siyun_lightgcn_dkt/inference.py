import os
import argparse

import numpy as np
import torch

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    # feat siyun : add logger
    logger = get_logger(logging_conf)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args=args)
    preprocess.load_test_data(file_name=args.test_file_name)
    
    # feat siyun : add user/item test emb path
    user2index = preprocess.user2index
    item2index = preprocess.item2index
    
    test_data: np.ndarray = preprocess.get_test_data()
    
    
    
    
    logger.info("Loading Model ...")
    # feat siyun : add get_model
    ## user2index, item2index를 따로 받지않고 get_model에서 사용된 것을 사용. 
    # model = trainer.load_model(args=args, get_model=trainer.get_model, user2index, item2index).to(args.device)
    model = trainer.load_model(args, trainer.get_model, user2index, item2index).to(args.device)

    
    logger.info("Make Predictions & Save Submission ...")
    #feat siyun : add user/item_test_emb
    trainer.inference(args=args, test_data=test_data, model=model)

    
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
