# from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
import os
import random
import warnings
import sys

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}

data_dir = '/opt/ml/data/'
df = pd.read_csv(data_dir + 'train_data.csv', dtype=dtype, parse_dates=['Timestamp'])
df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

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

    df['test_type'] = df['assessmentItemID'].apply(test_type)
    df['paper_type'] = df['assessmentItemID'].apply(paper_type).astype(int)
    df['paper_subtype'] = df['assessmentItemID'].apply(paper_subtype).astype(int)
    
    df['paper_number_percent'] = df.groupby('paper_number')['answerCode'].transform(avg_percent)
    df['paper_type_percent'] = df.groupby('paper_type')['answerCode'].transform(avg_percent)
    df['KnowledgeTag_percent'] = df.groupby('KnowledgeTag')['answerCode'].transform(avg_percent)

    return df

def check_components(df) :
    # kmeans로 몇개의 그룹이 최적화 대상이 될 수 있는지 확인합니다.
    # 여기서는 태그를 기준으로 묶음.
    x = df[['KnowledgeTag']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)

    inertia = []

    Km = range(1,50)
    for k in Km :
        kmeans = KMeans(n_clusters = k, random_state=42)
        kmeans.fit(scaled)
        inertia.append(kmeans.inertia_)

    # plt.figure(figsize=(10, 6))
    # plt.plot(Km, inertia, 'bx-')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.title('The Elbow Method showing the optimal K')
    # plt.show()
    return Km, inertia

def pca(df) :
    # svd를 진행할 대상을 고르기
    x = df[['KnowledgeTag']]
    
    # one hot encoding을 통해 multi hot encoding 진행
    encoder = OneHotEncoder()
    x_encode = encoder.fit_transform(x)
    
    # 결과를 확인하고 싶으면
    # xx = x_encode.toarray()
    
    # k-means로 대략적으로 확인한 componenets를 바탕으로 설정
    # svd = TruncatedSVD(n_components=3)
    # encode_pca = svd.fit_transform(x_encode)
    
    # df['KnowledgeTag_SVD_1'] = encode_pca[:, 0] # 5개 주성분 중 첫 번째 주성분
    # df['KnowledgeTag_SVD_2'] = encode_pca[:, 1] # 5개 주성분 중 두 번째 주성분
    # df['KnowledgeTag_SVD_3'] = encode_pca[:, 2] # 5개 주성분 중 세 번째 주성분
    
    pca = PCA(n_components=1)
    df['KnowledgeTag_pca'] = pca.fit_transform(x)
    return df
    
def total_input(df) :
    # SVD를 바탕으로 재해석한 Tag를 추가적인 input 생성
    # train data에 적용시킨 것이기 때문에 추가적인 구조의 변경이 필요.
    df['elapsed'] = df['elapsed'].fillna(0)

    X = df[['KnowledgeTag','paper_number','paper_type','paper_subtype',
         'elapsed','cumulative','paper_number_percent',
         'paper_type_percent','KnowledgeTag_percent','elapsed_pca']]

    return X


# 결과를 여기에서 확인해보고 싶으면 밑의 명령어 입력
# df = elapsed(df)
# df = cumsum(df)
# df = type_percent(df)
# df = pca(df)
# df = total_input(df)


# run this code if you want to know how many clusters you need for decompositioning.
# check_components(df)
