import numpy as np
import pandas as pd

def preprocessing(df : pd.DataFrame) -> pd.DataFrame:
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)

    #순간정답 수
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df = df.fillna(0) # 결측치 0으로 처리
    #유저들의 순간문제 풀이수
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    #순간정답률 feature
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    df = df.fillna(0) # 결측치 0으로 처리
    #elapsed feature
    diff = df.loc[:, ['userID','testId', 'Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
    df['elapsed'] = diff
    ####이전행으로 하나씩 밀어주는 작업 필요할지 논의 필요 ## 
    ####df['elapsed'] = diff.shift(-1)

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

    #df.fillna(0,inplace=True) # 결측치 0으로 처리 # 어떤 값으로 처리할 지 생각
    return df

def correct_shift_future(df : pd.DataFrame, window = 2) -> pd.DataFrame:
    #미래의 특정 시점에 문제 정답 맞춤 여부
    ######## 일단 shift를 2까지 줬는데 추후 범위를 결정 예정
    ### 미래 정보
    for i in range(1,window+1,1):
        df[f'correct_shift_-{i}'] = df.groupby('userID')['answerCode'].shift(i*(-1))

    #df.fillna(0,inplace=True) # 결측치 0으로 처리 # 어떤 값으로 처리할 지 생각
    return df