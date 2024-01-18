import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

## 수치형 Feature 만들기
# 유저별 knoledgetag 개수
def feat_knowledgetag_count(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    knowledgetag_count = df.groupby('userID')['KnowledgeTag'].nunique().fillna(0)
    df['knowledgetag_count'] = df['userID'].map(knowledgetag_count)
    
    return df

# 유저별 testId의 고유 개수
def feat_testId_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    testId_count = df.groupby("userID")["testId"].nunique().fillna(0)
    df['testId_count'] = df['userID'].map(testId_count)

    return df

# 유저별 assessmentItemID의 고유 개수
def feat_assessmentItemID_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assessmentItemID_count = df.groupby('userID')['assessmentItemID'].nunique().fillna(0)
    df['assessmentItemID_count'] = df['userID'].map(assessmentItemID_count)

    return df

# knoledgetag 별 assessmentItemID의 고유 개수
def feat_assessmentItemID_per_knowledgetag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assessmentItemID_per_testId = df.groupby('KnowledgeTag')['assessmentItemID'].nunique().fillna(0)
    df['assessmentItemID_per_testId_count'] = df['testId'].map(assessmentItemID_per_testId)

    return df

# testId 별 assessmentItemID의 고유 개수
def feat_assessmentItemID_per_testId(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assessmentItemID_per_testId = df.groupby('testId')['assessmentItemID'].nunique().fillna(0)
    df['assessmentItemID_per_testId_count'] = df['testId'].map(assessmentItemID_per_testId)

    return df


## 누적 Feature 만들기
# 사용자별 testId 별 누적으로 푼 문제(assessmentItemID) 수
def feat_user_ass_cumcount_per_testId(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['user_ass_cumcount_per_testId'] = df.groupby(['userID', 'testId'])['assessmentItemID'].cumcount()

    return df

# 사용자별 testId 별 누적 정답 수
def feat_user_ass_cumsum_per_testId(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['user_ass_cumsum_per_testId'] = df.groupby(['userID', 'testId'])['answerCode'].cumsum().shift(fill_value=0)

    return df

# 사용자별 testId 별 누적 정답률
def feat_user_ass_per_testId_average(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 이전 문제 정답 횟수
    df['user_ass_cumsum_per_testId'] = df.groupby(['userID', 'testId'])['answerCode'].cumsum().shift(fill_value=0)
    
    # 누적 문제 수
    df['user_ass_cumcount_per_testId'] = df.groupby(['userID', 'testId'])['assessmentItemID'].cumcount()
    
    # 누적 정답률
    df['user_ass_per_testId_average'] = df['user_ass_cumsum_per_testId'] / df['user_ass_cumcount_per_testId']
    df['user_ass_per_testId_average'] = df['user_ass_per_testId_average'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.drop('shift',axis =1)
    
    return df

# 사용자별 KnowledgeTag 별 누적으로 푼 문제(assessmentItemID) 수
def feat_user_ass_cumcount_per_knowledgetag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['user_ass_cumcount_per_KnowledgeTag'] = df.groupby(['userID', 'KnowledgeTag'])['assessmentItemID'].cumcount()

    return df
# 사용자별 KnowledgeTag 별 누적으로 푼 누적 정답수 
def feat_user_ass_cumsum_per_knowledgetag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['user_ass_cumsum_per_KnowledgeTag'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].cumsum().shift(fill_value=0)

    return df

# 사용자별 KnowledgeTag 별 누적 정답률
def feat_user_ass_per_knowledgetag_average(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 이전 문제 정답 횟수
    df['user_ass_cumsum_per_KnowledgeTag'] = df.groupby(['userID', 'testId'])['answerCode'].cumsum().shift(fill_value=0)
    
    # 누적 문제 수
    df['user_ass_cumcount_per_KnowledgeTag'] = df.groupby(['userID', 'testId'])['assessmentItemID'].cumcount()
    
    # 누적 정답률
    df['user_ass_per_KnowledgeTag_average'] = df['user_ass_cumsum_per_nowledgeTag'] / df['user_ass_cumcount_per_KnowledgeTag']
    df['user_ass_per_KnowledgeTag_average'] = df['user_ass_per_KnowledgeTag_average'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.drop('shift',axis =1)
    
    return df

# 과거에 해당 knowledgetag를 맞춘 누적 횟수 <<  sehoon 6. 태그별 누적으로 푼 문제 수와 연관(feat_tag_cumsum)
def feat_past_knowledgetag_correct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['shift'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].shift().fillna(0)
    df['past_knowledgetag_correct'] = df.groupby(['userID', 'KnowledgeTag'])['shift'].cumsum()
    df = df.drop('shift',axis =1)

    return df


# 과거에 해당 knowledgetag를 풀었던 누적 횟수 << 6. 태그별 누적으로 푼 문제 수 (tag_cumsum) 와 중복되는걸 확인
def feat_past_knowledgetag_attempts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['past_knowledgetag_attempts'] = df.groupby(['userID', 'KnowledgeTag']).cumcount().fillna(0)

    return df

# 과거에 풀었던 해당 knowledgetag에 대한 누적 정답률
def feat_past_knowledgetag_average(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['shift'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].shift().fillna(0)
    df['past_knowledgetag_correct'] = df.groupby(['userID', 'KnowledgeTag'])['shift'].cumsum()
    df['past_knowledgetag_attempts'] = df.groupby(['userID', 'KnowledgeTag']).cumcount().fillna(0)
    df['past_knowledgetag_average'] = df['past_knowledgetag_correct'] / df['past_knowledgetag_attempts']
    df['past_knowledgetag_average'] = df['past_knowledgetag_average'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.drop('shift',axis =1)
    
    return df

## 시간 관련 Feature 만들기

# 분 단위의 elapsed
def feat_elapsed_cumsum_minutes(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)

    df['elapsed_minutes'] = df['elapsed'] // 60

    return df




# test_id 별 문제 푸는데 사용한 시간 정규화
def feat_normalized_bytestId_elapsed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    df['normalized_bytestId_elapsed'] = df.groupby('testId')['elapsed'].transform(lambda x: (x - x.mean()) / x.std())
    
    return df

# knowledgetag 별 문제 푸는데 사용한 시간 정규화
def feat_normalized_byknowledgetag_elapsed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    df['normalized_byknowledgetag_elapsed'] = df.groupby('KnowledgeTag')['elapsed'].transform(lambda x: (x - x.mean()) / x.std())
    
    return df

'''
# 유저별 test_id 별 문제 푸는데 사용한 시간 정규화
def feat_normalized_usertestId_elapsed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    df['normalized_usertestId_elapsed'] = df.groupby(['userID', 'testId'])['elapsed'].transform(lambda x: (x - x.mean()) / x.std())
    
    return df

# 유저별 knowledgetag 별 문제 푸는데 사용한 시간 정규화
def feat_normalized_userknowledgetag_elapsed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    df['normalized_userknowledgetag_elapsed'] = df.groupby(['userID', 'KnowledgeTag'])['elapsed'].transform(lambda x: (x - x.mean()) / x.std())
    
    return df
'''