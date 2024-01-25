import os
import sys
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from boosting.utils import get_logger, set_seeds, logging_conf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/Feature_Engineering')
import sehoon.feat_eng_sehoon as sehoon

logger = get_logger(logging_conf)

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, df):
        
        data = {}
        users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
        random.shuffle(users)
        
        max_train_data_len = ((1 - self.args.test_size)*len(df))
        sum_of_train_data = 0
        user_ids =[]
        
        for user_id, count in users:
            sum_of_train_data += count
            if max_train_data_len < sum_of_train_data:
                break
            user_ids.append(user_id)

        train = df[df['userID'].isin(user_ids)]
        test = df[df['userID'].isin(user_ids) == False]

        # test데이터셋은 각 유저의 마지막 interaction만 추출
        test = test.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        test = test[test['userID'] != test['userID'].shift(-1)]
        
        data['X_train'] = train[self.args.X_columns]
        data['y_train'] = train[self.args.y_column]
        data['X_valid'] = test[self.args.X_columns]
        data['y_valid'] = test[self.args.y_column]
        
        logger.info(f"  Train : {len(data['X_train'])}, Valid : {len(data['X_valid'])}")
        
        return data
    
    def kfold_split_data(self, df):
        
        logger.info(f"  Kfold : divide into {self.args.n_fold}")
        kfold = KFold(self.args.n_fold, random_state=self.args.seed, shuffle=True)
        users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
        
        data_list = []
        max_train_data_len = (1 - 1/self.args.n_fold) * len(df)
        for fold_id, (train_ids, valid_ids) in enumerate(kfold.split(users)):
            fold_users = [users[ids] for ids in train_ids]
        
            sum_of_train_data = 0
            user_ids =[]
            for user_id, count in fold_users:
                sum_of_train_data += count
                if max_train_data_len < sum_of_train_data:
                    break
                user_ids.append(user_id)

            train = df[df['userID'].isin(user_ids)]
            test = df[df['userID'].isin(user_ids) == False]

            #test데이터셋은 각 유저의 마지막 interaction만 추출
            test = test.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
            test = test[test['userID'] != test['userID'].shift(-1)]
            
            data = {}
            data['X_train'] = train[self.args.X_columns]
            data['y_train'] = train[self.args.y_column]
            data['X_valid'] = test[self.args.X_columns]
            data['y_valid'] = test[self.args.y_column]
            data_list.append(data)
            
            logger.info(f"    Fold {fold_id + 1} > Train : {len(data['X_train'])}, Valid : {len(data['y_valid'])}")
        
        return data_list

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        
        setattr(self.args, 'columns', list(df.columns))
        setattr(self.args, 'X_columns', [column for column in self.args.columns if column not in [self.args.y_column, 'Timestamp']])

        # for col in cate_cols:
        for col in ['assessmentItemID', 'testId']:
            df[col] = df[col].astype('category').cat.codes
        
        # ⚠️주의
        # Feature Engineering 데이터는 모두 숫자 데이터라 가정하고 별도 조치 안함.
        
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        
        df = sehoon.feat_user_correct_stats(df, 'mean')
        # df = sehoon.feat_ass_correct_stats(df, 'mean')
        # df = sehoon.feat_testid_correct_stats(df, 'mean')
        # df = sehoon.feat_tag_correct_stats(df, 'mean')
        # df = sehoon.feat_user_ass_cumcount(df)
        # df = sehoon.feat_user_answer_cumsum(df)
        # df = sehoon.feat_user_answer_acc_per(df)
        # df = sehoon.feat_reverse_answer_cumsum(df)
        # df = sehoon.feat_testid_cumsum(df)
        # df = sehoon.feat_tag_cumsum(df)
        # df = sehoon.feat_relative_answer_score(df)
        # df = sehoon.feat_elapsed(df)
        # df = sehoon.feat_elapsed_cumsum(df)
        # df = sehoon.feat_normalized_elapsed(df)
        # df = sehoon.feat_elapsed_type_stats(df, 'mean')
        # df = sehoon.feat_relative_elapsed_time(df)
        # df = sehoon.feat_elapsed_pca(df)
        # df = sehoon.feat_elapsed_lda(df)
        # df = sehoon.feat_rolling_mean_time(df, 3)
        
        return df

    def load_data_from_file(self, train_file_name: str, test_file_name: str) -> np.ndarray:
        
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
        }
        train_file_path = os.path.join(self.args.data_dir, train_file_name)
        test_file_path = os.path.join(self.args.data_dir, test_file_name)
        train_df = pd.read_csv(train_file_path, dtype=dtype, parse_dates=['Timestamp'])
        test_df = pd.read_csv(test_file_path, dtype=dtype, parse_dates=['Timestamp'])
        
        # train, test 데이터 merge 사용
        merge_df = pd.concat([train_df, test_df])
        merge_df = merge_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        merge_df = self.__feature_engineering(merge_df)
        merge_df = self.__preprocessing(merge_df)
        
        train_df = merge_df[merge_df[self.args.y_column] != -1]
        test_df = merge_df[merge_df[self.args.y_column] == -1]
        
        return train_df, test_df

    def load_file(self, train_file_name: str, test_file_name: str) -> None:
        self.train_data, self.test_data = self.load_data_from_file(train_file_name, test_file_name)