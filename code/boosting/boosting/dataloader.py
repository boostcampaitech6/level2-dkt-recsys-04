import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/feture_engineering')
import feat_eng_sehoon as sehoon

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
        X = df[self.args.X_columns]
        y = df[self.args.y_column]
        
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = train_test_split(
                                                              X,
                                                              y,
                                                              test_size=self.args.test_size,
                                                              random_state=self.args.seed,
                                                              shuffle=self.args.data_shuffle
                                                             )
        
        test_data = pd.merge(data['X_valid'], data['y_valid'], left_index=True, right_index=True, how='inner').sort_index()
        # test데이터셋은 각 유저의 마지막 interaction만 추출 (예측해야 할 데이터는 마지막 데이터임)
        test_data = test_data[test_data['userID'] != test_data['userID'].shift(-1)]
        data['X_valid'] = test_data[self.args.X_columns]
        data['y_valid'] = test_data[self.args.y_column]
        
        return data

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        
        # Timestamp는 Feature Engineering 했다고 보고 삭제 조치
        df = df.drop('Timestamp', axis=1)
        
        setattr(self.args, 'columns', list(df.columns))
        setattr(self.args, 'y_column', 'answerCode')
        setattr(self.args, 'y_index', self.args.columns.index(self.args.y_column))
        setattr(self.args, 'X_columns', [column for column in self.args.columns if column not in [self.args.y_column]])

        # for col in cate_cols:
        for col in ['assessmentItemID', 'testId']:
            df[col] = df[col].astype('category').cat.codes
        
        # ⚠️주의
        # Feature Engineering 데이터는 모두 숫자 데이터라 가정하고 별도 조치 안함.
        
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        
        # df = sehoon.feat_elapsed(df)
        # df = sehoon.feat_elapsed_cumsum(df)
        # df = sehoon.feat_ass_correct_stats(df)
        df = sehoon.feat_user_answer_acc_per(df)
        # df = sehoon.feat_relative_answer_score(df)
        # df = sehoon.feat_normalized_elapsed(df)
        # df = sehoon.feat_relative_elapsed_time(df)
        
        return df

    def load_data_from_file(self, file_name: str, test_bool: bool=False) -> np.ndarray:
        
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
        }
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, dtype=dtype, parse_dates=['Timestamp'])
        
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df)
        
        # 예측 해야 하는 데이터만 필터
        if test_bool:
            df = df[df[self.args.y_column] == -1]
        
        return df

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, test_bool=True)