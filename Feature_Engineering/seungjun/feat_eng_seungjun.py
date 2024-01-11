import numpy as np
import pandas as pd

def preprocessing(df : pd.DataFrame) -> pd.DataFrame:
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)

    #outlier
    ##전체정답률 1인 유저 삭제
    user_acc_1 = list(df.groupby('userID').user_acc.last()[df.groupby('userID').user_acc.last() == 1].index)
    df = df[~df['userID'].isin(user_acc_1)]

    ##전체정답률 0인 유저 삭제
    user_acc_0 = list(df.groupby('userID').user_acc.last()[df.groupby('userID').user_acc.last() == 0].index)
    df = df[~df['userID'].isin(user_acc_0)]

    return df

def correct_shift_past(df : pd.DataFrame, window = 2) -> pd.DataFrame:

    #과거의 특정 시점에 문제 정답 맞춤 여부
    ######## 일단 shift를 2까지 줬는데 추후 범위를 결정 예정
    ### 과거 정보
    for i in range(1,window+1,1):
        df[f'correct_shift_{i}'] = df.groupby('userID')['answerCode'].shift(i)

    df.fillna(0,inplace=True) # 결측치 -1로 처리 # 0이나 1로 결측치를 처리할 수 없음
    return df

def correct_shift_future(df : pd.DataFrame, window = 2) -> pd.DataFrame:
    #미래의 특정 시점에 문제 정답 맞춤 여부
    ######## 일단 shift를 2까지 줬는데 추후 범위를 결정 예정
    ### 미래 정보
    for i in range(1,window+1,1):
        df[f'correct_shift_-{i}'] = df.groupby('userID')['answerCode'].shift(i*(-1))

    df.fillna(0,inplace=True) # 결측치 -1으로 처리 # 0이나 1로 결측치를 처리할 수 없음
    return df