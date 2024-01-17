import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ################## 여기서 data 처리 이우러짐 #####################
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name) # 데이터 로드한 것을 none이었던 self.train_data에 정의
    train_data: np.ndarray = preprocess.get_train_data() # self.train_data을 반환
    train_data, valid_data = preprocess.split_data(data=train_data)
    wandb.init(project="dkt", config=vars(args))
    #################################################################

    # 모델 선택
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    
    # 학습 실행
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)

# Python 인터프리터는 스크립트를 실행할 때 __name__을 "__main__"으로 설정
if __name__ == "__main__":
    args = parse_args() # 인자들 저장
    
    os.makedirs(args.model_dir, exist_ok=True)
    main(args) # main함수 실행 -> 학습!!!!!!!!!!!
