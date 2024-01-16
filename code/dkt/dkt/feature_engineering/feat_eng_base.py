import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def elapsed(df) :
    diff_train = df.loc[:, ['userID','Timestamp']].groupby('userID').diff().shift(-1)
    diff_train = diff_train['Timestamp'].apply(lambda x : x.total_seconds())
    df['elapsed'] = diff_train
    
    df.groupby('userID').apply(lambda x :x.iloc[:-1])

    # 한 시간이 지나면 outlier로 처리
    outlier = 1*3600
    non_outlier = df[df['elapsed'] <= outlier]
    # outlier에 해당하지 않는 row로 재구성 한 후 각 태그의 평균처리
    mean_elapsed = non_outlier.groupby('KnowledgeTag')['elapsed'].mean()
    df.loc[df['elapsed'] > outlier, 'elapsed'] = df[df['elapsed'] > outlier].apply(lambda x: mean_elapsed.get(x['KnowledgeTag'], x['elapsed']), axis=1)
    
    return df


def avg_percent(x) :
    return np.sum(x) / len(x)
    
def test_type(x) :
    # 전부 A로 동일
    return  x[0]
def paper_type(x) :
    # 0~9로 시험지 대분류로 가정
    return x[2]
def paper_subtype(x) :
    # ~~ 시험지 중분류로 가정
    return x[4:7]

def type_percent(df) :
    # 위에서 처리한 type을 변환하여 각각의 정답률 처리

    # df['test_type'] = df['assessmentItemID'].apply(test_type)
    # df['paper_type'] = df['assessmentItemID'].apply(paper_type).astype(int)
    # df['paper_subtype'] = df['assessmentItemID'].apply(paper_subtype).astype(int)
    
    # df['paper_number_percent'] = df.groupby('paper_number')['answerCode'].transform(avg_percent)
    # df['paper_type_percent'] = df.groupby('paper_type')['answerCode'].transform(avg_percent)
    df['KnowledgeTag_percent'] = df.groupby('KnowledgeTag')['answerCode'].transform(avg_percent)

    return df

def cumsum(df) :
    # 누적합
    _cumsum = df.loc[:, ['userID', 'answerCode']].groupby('userID').agg({'answerCode': 'cumsum'})
    # 누적갯수
    _cumcount = df.loc[:, ['userID', 'answerCode']].groupby('userID').agg({'answerCode': 'cumcount'}) + 1

    cum_ans = _cumsum / _cumcount
    df['cumulative'] = cum_ans['answerCode']

    df['paper_number'] = df['assessmentItemID'].apply(lambda x: x[7:]) # assessmentItemID의 뒤에 3자리를 의미 -> 각 시험지 별로 문제번호
    # item 열을 int16으로 변경
    df["paper_number"] = df["paper_number"].astype("int16")
    
    return df