import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/feture_engineering')
import fe_siyun as siyun # 이 부분 수정 필요
#from .feature_engineering import feat_eng_base  
from torch.nn.utils.rnn import pad_sequence

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

class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        
        test, question, tag, correct, elapsed, KnowledgeTag_percent, cumulative, paper_number = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]

        cate_cols = [test, question, tag, correct]
        cont_cols = [elapsed, KnowledgeTag_percent, cumulative, paper_number]
        
        if seq_len > self.args.max_seq_len:
            # cate_col
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
            # cont_col
            for i, col in enumerate(cont_cols):
                cont_cols[i] = col[-self.args.max_seq_len:]
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[:seq_len] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        cate_cont_cols = []
        cate_cont_cols.extend(cate_cols)
        cate_cont_cols.extend(cont_cols)
        
        for i, col in enumerate(cate_cont_cols):
            cate_cont_cols[i] = torch.tensor(col)
        return cate_cont_cols

    def __len__(self):
        return len(self.data)

#padding을 위한 함수
def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    # print("column 개수",col_n)
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            col_list[i].append(col)

    # 각 column의 값들을 대상으로 padding 진행
    # pad_sequence([[1, 2, 3], [3, 4]]) -> [[1, 2, 3],
    #                                       [3, 4, 0]]
    for i, col_batch in enumerate(col_list):
        col_list[i] = pad_sequence(col_batch, batch_first=True)

    # mask의 경우 max_seq_len을 기준으로 길이가 설정되어있다.
    # 만약 다른 column들의 seq_len이 max_seq_len보다 작다면
    # 이 길이에 맞추어 mask의 길이도 조절해준다
    col_seq_len = col_list[0].size(1)
    mask_seq_len = col_list[-1].size(1)
    if col_seq_len < mask_seq_len:
        col_list[-1] = col_list[-1][:, :col_seq_len]

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False

    trainset = DKTDataset(train, args)
    valset = DKTDataset(valid, args)

    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                               batch_size=args.batch_size,
                                               pin_memory=pin_memory,
                                               collate_fn=collate)

    valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                               batch_size=args.batch_size,
                                               pin_memory=pin_memory,
                                               collate_fn=collate)

    return train_loader, valid_loader

# 배치 전처리
def process_batch(batch, args):

    test, question, tag, correct, mask, elapsed, KnowledgeTag_percent, cumulative, paper_number = batch 
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #  saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1) #dim 1에 해당하는 값을 1씩 이동
    interaction[:, 0] = 0 # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)


    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)


    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    #cont 추가
    elapsed = elapsed.to(torch.float32)
    KnowledgeTag_percent = KnowledgeTag_percent.to(torch.float32)
    cumulative = cumulative.to(torch.float32)
    paper_number = paper_number.to(torch.float32)

    elapsed = elapsed.to(args.device)
    KnowledgeTag_percent = KnowledgeTag_percent.to(args.device)
    cumulative = cumulative.to(args.device)
    paper_number = paper_number.to(args.device)

    return (test, question,
            tag, correct, mask, 
            elapsed, KnowledgeTag_percent, cumulative, paper_number,
            interaction, gather_index)
