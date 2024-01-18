import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


# feature_engineering
from .feature_engineering import feat_eng_base 
from .feature_engineering import feat_eng_chanwoo
from .feature_engineering import feat_eng_sehoon

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    # 카테고리 변수 라벨 인코딩
    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        cate_cols = self.args.cat_cols
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy") # 개수를 빼둠
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df
    


    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        df = feat_eng_sehoon.feat_user_correct_stats(df, 'mean')
        df = feat_eng_sehoon.feat_ass_correct_stats(df, 'mean')
        df = feat_eng_sehoon.feat_testid_correct_stats(df, 'mean')
        df = feat_eng_sehoon.feat_tag_correct_stats(df, 'mean')
        df = feat_eng_sehoon.feat_user_ass_cumcount(df)
        df = feat_eng_sehoon.feat_user_answer_cumsum(df)
        df = feat_eng_sehoon.feat_user_answer_acc_per(df)
        df = feat_eng_sehoon.feat_reverse_answer_cumsum(df)
        df = feat_eng_sehoon.feat_testid_cumsum(df)
        df = feat_eng_sehoon.feat_tag_cumsum(df)
        df = feat_eng_sehoon.feat_relative_answer_score(df)
        df = feat_eng_sehoon.feat_elapsed(df)
        df = feat_eng_sehoon.feat_elapsed_cumsum(df)
        df = feat_eng_sehoon.feat_normalized_elapsed(df)
        df = feat_eng_sehoon.feat_elapsed_type_stats(df, 'mean')
        df = feat_eng_sehoon.feat_relative_elapsed_time(df)
        df = feat_eng_sehoon.feat_elapsed_pca(df)
        df = feat_eng_sehoon.feat_elapsed_lda(df)
        df = feat_eng_sehoon.feat_rolling_mean_time(df, 3)
        ## 추가
        return df

    # csv data 로드 + feature engineering + 정리 -> 튜플로 반환
    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        dtype = {'userID': 'int16', 
                 'answerCode': 'int8', 
                 'KnowledgeTag': 'int16'}
        csv_file_path = os.path.join(self.args.data_dir, file_name) 
        # dytype하고 parse_dates=['Timestamp']추가하고 위에 convert_time주석
        df= pd.read_csv(csv_file_path, dtype=dtype, parse_dates=['Timestamp']) # , nrows=100000)
        df = self.__feature_engineering(df)
        print(df.columns)
        df = self.__preprocessing(df, is_train)

        # [건우] category변수의 unique의 개수를 args에 등록 -> 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용 
        for col in self.args.cat_cols:
            setattr(self.args, f'n_{col}', len(np.load(os.path.join(self.args.asset_dir, f'{col}_classes.npy'))))
        

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = self.args.cat_cols + self.args.con_cols + self.args.tgt_col # [건우] 자동화 코드(추가)
        group = (
            df
            .groupby("userID")
            .apply(
                lambda r: tuple(r[col].values for col in columns) # [건우] 자동화 코드(추가)
                )
            )
        return group.values 

    # (csv data 로드 + feature engineering + 정리)한 것을 none이었던 self.train_data에 정의
    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name) 

    # (csv data 로드 + feature engineering + 정리)한 것을 none이었던 self.test_data에 정의
    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False) 


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.max_seq_len = args.max_seq_len
        self.args = args

    # 주어진 인덱스 index(row)에 해당하는 데이터를 반환하는 메서드
    def __getitem__(self, index: int) -> dict:
        row = self.data[index] 
        
        # [건우] 자동화 코드(추가)
        data={}
        columns = self.args.cat_cols + self.args.con_cols + self.args.tgt_col
        for i, col in enumerate(columns):
            if i < len(self.args.cat_cols): # categorical
                data[col] = torch.tensor(row[i] + 1, dtype=torch.int) # embedding 때문에 0과 구분하려고 +1함
            else: # continous
                data[col] = torch.tensor(row[i], dtype=torch.float) 
        

        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
        
        # Generate interaction(새로 생성) : 이전 sequence를 학습하기 위한 작업
        interaction = data["answerCode"] + 1  # 패딩을 위해 correct값에 1을 더해준다.(패딩이 0이기 때문)
        # roll(shifts=1): 텐서(또는 배열)를 한 칸씩 오른쪽으로 이동 ex) [a, b, c, d] -> [d, a, b, c]
        interaction = interaction.roll(shifts=1) 
        interaction_mask = data["mask"].roll(shifts=1) 
        interaction_mask[0] = 0 # 오른쪽으로 한 칸 옮겨서 상호작용 계산하는데 첫 번째는 이전 sequence가 없어서 0넣음
        interaction = (interaction * interaction_mask).to(torch.int64) # 상호작용 계산
        data["interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        # torch.utils.data.DataLoader : model에 feed(batch, shuffle, cpu와 gpu변환)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader

## [찬우] Data Augumentaion 추가

def slidding_window(data, args):
    window_size = args.max_seq_len

    if args.stride is None:
        args.stride = args.max_seq_len

    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data
        