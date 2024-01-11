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

## 시간에 대한 정보 반영 feature
#과거의 특정 시점에 문제 정답 맞춤 여부
def feat_correct_shift_past(df : pd.DataFrame, window = 2) -> pd.DataFrame:
    #과거의 특정 시점에 문제 정답 맞춤 여부
    ######## 일단 shift를 2까지 줬는데 추후 범위를 결정 예정
    ### 과거 정보
    for i in range(1,window+1,1):
        df[f'correct_shift_{i}'] = df.groupby('userID')['answerCode'].shift(i)

    df.fillna(0,inplace=True) # 결측치 -1로 처리 # 0이나 1로 결측치를 처리할 수 없음
    return df

#미래의 특정 시점에 문제 정답 맞춤 여부
def feat_correct_shift_future(df : pd.DataFrame, window = 2) -> pd.DataFrame:
    ######## 일단 shift를 2까지 줬는데 추후 범위를 결정 예정
    ### 미래 정보
    for i in range(1,window+1,1):
        df[f'correct_shift_-{i}'] = df.groupby('userID')['answerCode'].shift(i*(-1))

    df.fillna(0,inplace=True) # 결측치 -1으로 처리 # 0이나 1로 결측치를 처리할 수 없음
    return df

# 문제 푸는 시간대별 정답률
def feat_correct_per_hour_user_content(df : pd.DataFrame) -> pd.DataFrame:
    # 문제를 푸는 시간대
    df['hour'] = df['Timestamp'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)

    # 시간대별 정답률
    hour_dict = df.groupby(['hour'])['answerCode'].mean().to_dict()
    df['correct_per_hour'] = df['hour'].map(hour_dict)

    # 시간대별 특정유저 정답률
    user_hour_dict = df.groupby(['userID','hour'])['answerCode'].mean().to_dict()
    df['correct_per_user_hour'] = df.apply(lambda x: user_hour_dict.get((x['userID'], x['hour'])), axis=1)
    
    # 시간대별 특정문제 정답률
    content_hour_dict = df.groupby(['assessmentItemID','hour'])['answerCode'].mean().to_dict()
    df['correct_per_content_hour'] = df.apply(lambda x: content_hour_dict.get((x['assessmentID'], x['hour'])), axis=1)

    return df