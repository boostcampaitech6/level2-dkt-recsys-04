import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import Saint
from .optimizer import get_optimizer
from .scheduler import get_scheduler
# from .utils import get_logger, logging_conf

# logger = get_logger(logger_conf=logging_conf)

def run(args, train_data, valid_data, gradient=False):

    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    # augmentation
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

    train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # 🌟 분석에 사용할 값 저장 🌟
    report = OrderedDict()

    # gradient step 분석에 사용할 변수
    if gradient:
        args.n_iteration = 0
        args.gradient = OrderedDict()

        # 모델의 gradient값을 가리키는 모델 명 저장
        args.gradient['name'] = [name for name, _ in model.named_parameters()]

    best_auc = -1
    best_auc_epoch = -1
    best_acc = -1
    best_acc_epoch = -1
    for epoch in notebook.tqdm(range(args.n_epochs)):
        epoch_report = {}

        ### TRAIN
        train_start_time = time.time()
        train_auc, train_acc = train(train_loader, model, optimizer, scheduler, args, gradient)
        train_time = time.time() - train_start_time

        epoch_report['train_auc'] = train_auc
        epoch_report['train_acc'] = train_acc
        epoch_report['train_time'] = train_time

        ### VALID
        valid_start_time = time.time()
        valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)
        valid_time = time.time() - valid_start_time

        epoch_report['valid_auc'] = valid_auc
        epoch_report['valid_acc'] = valid_acc
        epoch_report['valid_time'] = valid_time

        # save lr
        epoch_report['lr'] = optimizer.param_groups[0]['lr']


        # 🌟 save it to report 🌟
        report[f'{epoch + 1}'] = epoch_report


        ### TODO: model save or early stopping
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_auc_epoch = epoch + 1

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_acc_epoch = epoch + 1

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)

    # save best records
    report['best_auc'] = best_auc
    report['best_auc_epoch'] = best_auc_epoch
    report['best_acc'] = best_acc
    report['best_acc_epoch'] = best_acc_epoch

    # save gradient informations
    if gradient:
        report['gradient'] = args.gradient
        del args.gradient
        del args['gradient']

    return report

def train(train_loader, model, optimizer, scheduler, args, gradient=False):
    model.train()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        
        preds = model(input)
        targets = input[3] # correct
        index = input[-1] # gather index
        
        loss = compute_loss(preds, targets, index)
        loss.backward()

        # save gradient distribution
        if gradient:
            args.n_iteration += 1
            args.gradient[f'iteration_{args.n_iteration}'] = get_gradient(model)

        # grad clip
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        optimizer.zero_grad()

        # warmup scheduler
        if args.scheduler == 'linear_warmup':
            scheduler.step()

        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    return auc, acc

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3] # correct
        index = input[-1] # gather index

        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    return auc, acc, total_preds, total_targets

# def inference(args, test_data: np.ndarray, model: nn.Module) -> None:
#     model.eval()
#     _, test_loader = get_loaders(args=args, train=None, valid=test_data)

#     total_preds = []
#     for step, batch in enumerate(test_loader):
#         batch = {k: v.to(args.device) for k, v in batch.items()}
#         preds = model(batch) # [건우] '**'를 사용하기 위해 parameter와 argument의 쌍이 같아햐 하는데 lstm에서 paramete는 data하나기 때문에 '**'안씀

#         # predictions
#         preds = sigmoid(preds[:, -1])
#         preds = preds.cpu().detach().numpy()
#         total_preds += list(preds)

#     write_path = os.path.join(args.output_dir, "submission.csv")
#     os.makedirs(name=args.output_dir, exist_ok=True)
#     with open(write_path, "w", encoding="utf8") as w:
#         w.write("id,prediction\n")
#         for id, p in enumerate(total_preds):
#             w.write("{},{}\n".format(id, p))
#     logger.info("Successfully saved submission as %s", write_path)

def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'last_query': model = LastQuery(args)
    if args.model == 'saint': model = Saint(args)
    if args.model == 'tfixup': model = FixupEncoder(args)

    model.to(args.device)

    return model

def compute_loss(preds, targets, index):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
        index    : (batch_size, max_seq_len)

        만약 전체 sequence 길이가 max_seq_len보다 작다면 해당 길이로 진행
    """
    loss = get_criterion(preds, targets)
    loss = torch.gather(loss, 1, index)
    loss = torch.mean(loss)

    return loss

def get_gradient(model):
    gradient = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad != None:
            gradient.append(grad.cpu().numpy().astype(np.float16))
            # gradient.append(grad.clone().detach())
        else:
            gradient.append(None)

    return gradient

