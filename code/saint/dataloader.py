import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/feture_engineering')
import fe_siyun as siyun

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None

    def get_train_data(self):
        return self.train_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.data_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df):
        
        #con_col에 대한 전처리
        con_cols= ["elapsed", "KnowledgeTag_percent", "cumulative", "paper_number"]
        df = siyun.elapsed(df)
        df = siyun.cumsum(df)
        df = siyun.type_percent(df)
        
        #cate_col에 대한 전처리
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
        
        #categorical data label encoding
        for col in cate_cols:

            #For UNKNOWN class
            a = df[col].unique().tolist() + [np.nan]

            le = LabelEncoder()
            le.fit(a)
            df[col] = le.transform(df[col])
            self.__save_labels(le, col)
        
        return df

    def load_data_from_file(self, file_name):
       
        dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
        }

        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path,dtype=dtype, parse_dates=['Timestamp'])
        df = self.__preprocessing(df)
        
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = df['assessmentItemID'].nunique()
        self.args.n_test = df['testId'].nunique()
        self.args.n_tag = df['KnowledgeTag'].nunique() 
        
        df = df.sort_values(by=['userID','Timestamp'], axis=0)

        #columns 추가
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag',"elapsed", "KnowledgeTag_percent", "cumulative", "paper_number"]

        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values,
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values,
                    r['elapsed'].values,
                    r['KnowledgeTag_percent'].values,
                    r['cumulative'].values,
                    r['paper_number'].values,
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)
