import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from torch_geometric.nn.models import LightGCN
import wandb
import argparse
from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)

def build(n_node: int, weight: str = None, **kwargs):

    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def run(    
    # feat siyun : add arg
    ## arg를 받아 처리합니다. 여기에서는 purpose를 사용
    args : argparse.Namespace,
    model: nn.Module,
    train_data: dict,
    valid_data: dict = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
):
    
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    os.makedirs(name=model_dir, exist_ok=True)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])
        
        # feat siyun : add features
        ## -------------
        # features = train_data['features']
        # label = label.to("cpu").detach().numpy()
        # valid_data = dict(edge=edge[:, eids], label=label[eids],
        #                   features = features[eids])
        ## -------------

        
        
        
    if args.purpose == 'result' :    
        logger.info(f"Training Started : n_epochs={n_epochs}")
        best_auc, best_epoch = 0, -1
        for e in range(n_epochs):
            logger.info("Epoch: %s", e)
            # TRAIN
            train_auc, train_acc, train_loss = train(train_data=train_data, model=model, optimizer=optimizer)
        
            # VALID
            auc, acc = validate(valid_data=valid_data, model=model)
            wandb.log(dict(train_loss_epoch=train_loss,
                        train_acc_epoch=train_acc,
                        train_auc_epoch=train_auc,
                        valid_acc_epoch=acc,
                        valid_auc_epoch=auc))

            if auc > best_auc:
                logger.info("Best model updated AUC from %.4f to %.4f", best_auc, auc)
                best_auc, best_epoch = auc, e
                torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                        f=os.path.join(model_dir, f"best_model.pt"))
                
        torch.save(obj={"model": model.state_dict(), "epoch": e + 1},
                f=os.path.join(model_dir, f"last_model.pt"))
        logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
    
    
    # feat siyun : add embedding arg parser
    ## purpose가 embedding인 경우 다음과 같은 처리를 진행합니다.
    ### https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/lightgcn.html
    ### https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py 다른 구조는 여기참고
    #### ----------------------------
    if args.purpose == 'embedding' :
        model.train()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        for e in range(n_epochs):
            train_auc, train_acc, train_loss = train(train_data=train_data, model=model, optimizer=optimizer)
            if valid_data is not None:
                auc, acc = validate(valid_data=valid_data, model=model)
        
        
        edge_index, edge_weight = train_data["edge"], train_data.get("edge_weight", None)
        embeddings = model.get_embedding(edge_index, edge_weight)
        # 결과를 embeddings.pt의 파이토치 파일로 저장
        torch.save(embeddings, os.path.join(model_dir, "embeddings.pt"))
        return embeddings
    #### ----------------------------
    
    


def train(model: nn.Module, train_data: dict, optimizer: torch.optim.Optimizer):
    
    pred = model(train_data["edge"])
    
    loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])
    
    prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_data["label"].cpu().numpy()
    acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
    auc = roc_auc_score(y_true=label, y_score=prob)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    logger.info("TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f", auc, acc, loss.item())
    return auc, acc, loss


def validate(valid_data: dict, model: nn.Module):
    with torch.no_grad():
        prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        prob = prob.detach().cpu().numpy()
        
        label = valid_data["label"]
        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(model: nn.Module, data: dict, output_dir: str):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)
        
    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
    logger.info("Successfully saved submission as %s", write_path)
