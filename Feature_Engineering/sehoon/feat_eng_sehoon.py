import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Feature Engineering

## 수치형 Feature 만들기
#### 1. 사용자별 정답에 대한 평균/중간값/표준편차 등
# mean, median, std
def feat_user_correct_stats(df : pd.DataFrame, static : str) -> pd.DataFrame:
    df = df.copy()
    col_name = 'cat_' + 'user_answer_' + static
    ass_cor_accuracy = df.groupby('userID')['answerCode'].agg([static])
    df[col_name] = df['userID'].map(ass_cor_accuracy[static])
    
    return df

#### 2. 문항별 정답에 대한 평균/중간값/표준편차 등
# mean, median, std
def feat_ass_correct_stats(df : pd.DataFrame, static : str) -> pd.DataFrame:
    df = df.copy()
    col_name = 'cat_' + 'ass_answer_' + static
    ass_cor_accuracy = df.groupby('assessmentItemID')['answerCode'].agg([static])
    df[col_name] = df['assessmentItemID'].map(ass_cor_accuracy[static])
    
    return df

#### 3. 시험지별 정답에 대한 평균/중간값/표준편차 등
# mean, median, std
def feat_testid_correct_stats(df : pd.DataFrame, static : str) -> pd.DataFrame:
    df = df.copy()
    col_name = 'cat_' + 'testid_answer_' + static
    ass_cor_accuracy = df.groupby('testId')['answerCode'].agg([static])
    df[col_name] = df['testId'].map(ass_cor_accuracy[static])
    
    return df

#### 4. 태그별 정답에 대한 평균/중간값/표준편차 등
# mean, median, std
def feat_tag_correct_stats(df : pd.DataFrame, static : str) -> pd.DataFrame:
    df = df.copy()
    col_name = 'cat_' + 'tag_answer_' + static
    ass_cor_accuracy = df.groupby('KnowledgeTag')['answerCode'].agg([static])
    df[col_name] = df['KnowledgeTag'].map(ass_cor_accuracy[static])

    return df

## 누적 Feature 만들기
#### 1. 사용자별 누적으로 푼 문제 수
def feat_user_ass_cumcount(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 누적 문제 수
    df['user_ass_cumcount'] = df.groupby('userID')['assessmentItemID'].cumcount()
    
    return df

#### 2. 사용자별 누적 정답 횟수
def feat_user_answer_cumsum(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 이전 문제 정답 횟수
    df['user_answer_cumsum'] = df.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)
    
    return df

#### 3. 사용자별 누적 정답률 구하기
def feat_user_answer_acc_per(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 이전 문제 정답 횟수
    df['user_answer_cumsum'] = df.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)
    
    # 누적 문제 수
    df['user_ass_cumcount'] = df.groupby('userID')['assessmentItemID'].cumcount()
    
    # 누적 정답률
    df['user_answer_acc_per'] = (df['user_answer_cumsum'] / df['user_ass_cumcount']).fillna(0)
    df = df.drop(['user_answer_cumsum', 'user_ass_cumcount'], axis=1)
    
    return df

#### 4. 사용자별 미래에 맞출 문제 수
def feat_reverse_answer_cumsum(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reverse_answer_cumsum = df.iloc[::-1].copy()
    
    # 미래에 맞출 문제 수
    reverse_answer_cumsum['answer_cumsum'] = reverse_answer_cumsum.groupby('userID')['answerCode'].cumsum().shift().fillna(0)
    df['reverse_answer_cumsum'] = reverse_answer_cumsum['answer_cumsum'].iloc[::-1]
    
    return df

#### 5. 시험지별 누적으로 푼 문제 수
def feat_testid_cumsum(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 태그별 푼 누적 문제 수
    df['testid_cumsum'] = df.groupby(['userID', 'testId']).cumcount()
    
    return df

#### 6. 태그별 누적으로 푼 문제 수
def feat_tag_cumsum(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 태그별 푼 누적 문제 수
    df['tag_cumsum'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()
    
    return df

## 상대적 Feature 만들기
#### 1. 정답률을 고려한 상대적 점수를 매겨 Feature에 반영 (문제 난이도)
def feat_relative_answer_score(df : pd.DataFrame) -> pd.DataFrame:
    """_summary_
    1) 문제를 맞았을 경우 (answered_correctly : 1)
    - 문제가 쉬울 경우 (accuracy_avg_by_content_id : 0.9) 결과는 `0.1`
    - 문제가 어려울 경우 (accuracy_avg_by_content_id : 0.2) 결과는 `0.8`
    2) 문제를 틀렸을 경우 (answered_correctly : 0)
    - 문제가 쉬울 경우 (accuracy_avg_by_content_id : 0.9) 결과는 `-0.9`
    - 문제가 어려울 경우 (accuracy_avg_by_content_id : 0.2) 결과는 `-0.2`
    - 값은 -1에서 +1 사이로 주어지며 학생의 상대적인 실력을 표현할 수 있는 feature다.
    """    
    df = df.copy()
    # 문항별 정답률
    ass_cor_accuracy = df.groupby('assessmentItemID')['answerCode'].mean()
    df['ass_cor_accuracy'] = df['assessmentItemID'].map(ass_cor_accuracy)
    
    # 난이도별 점수 부여
    df['ass_cor_accuracy'] = df.apply(lambda x: (1 - x['ass_cor_accuracy']) if x['answerCode'] == 1 else (-x['ass_cor_accuracy']), axis=1)
    
    # 정답률을 고려한 상대적 점수 누적 합
    df['relative_answer_score'] = df.groupby('userID')['ass_cor_accuracy'].cumsum().shift(fill_value=0)
    df = df.drop(['ass_cor_accuracy'], axis=1)
    
    return df

## 시간 관련 Feature 만들기
#### 1. 문제 푸는데 걸린 시간
def feat_elapsed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Timestamp -> 초단위 변환
    df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
    
    # 문제 푸는데 10분이상 걸리면 0으로 초기화
    df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)

    return df

#### 2. 문제 푸는데 걸린 누적 시간
def feat_elapsed_cumsum(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    df['elapsed_cumsum'] = df.groupby('userID')['elapsed'].cumsum()

    return df

#### 3. 사용자별 문제 푸는데 사용한 시간 정규화
def feat_normalized_elapsed(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    df['normalized_elapsed'] = df.groupby('userID')['elapsed'].transform(lambda x: (x - x.mean()) / x.std())
    
    return df

#### 4. 문제 푼 시간대별 정답률 평균/중간값/표준편차 등
# mean, median, std
def feat_elapsed_type_stats(df : pd.DataFrame, static : str) -> pd.DataFrame:
    """_summary_
    - 새벽 : 00 ~ 06 -> 1
    - 오전 : 06 ~ 12 -> 2
    - 오후 : 12 ~ 18 -> 3
    - 야간 : 18 ~ 00 -> 4
    """    
    df = df.copy()
    
    def AnswerTimeType(n : int):
        time_hour = n.hour
        if time_hour >= 18:
            return 4
        elif time_hour >= 12:
            return 3
        elif time_hour >= 6:
            return 2
        elif time_hour >= 0 :
            return 1

    col_name = 'cat_' + 'elapsed_type_' + static
    df['elapsed_type'] = df['Timestamp'].apply(AnswerTimeType)
    elapsed_type = df.groupby('elapsed_type')['answerCode'].agg([static])
    df[col_name] = df['elapsed_type'].map(elapsed_type[static])
    df = df.drop(['elapsed_type'], axis=1)
    
    return df

#### 5. 문제 푼 시간의 상대적 비교
def feat_relative_elapsed_time(df : pd.DataFrame) -> pd.DataFrame:
    """_summary_
    - 문제 풀이에 사용한 시간을 유저별 중간값과 차이를 통해 상대적으로 시작을 얼마나 썼는지 비교
    """    
    df = df.copy()
    if 'elapsed' not in df.columns:
        # Timestamp -> 초단위 변환
        df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
        df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
        
        # 문제 푸는데 10분이상 걸리면 0으로 초기화
        df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)

    df['relative_elapsed'] = df.groupby('userID')['elapsed'].transform(lambda x: x - x.median())
    
    return df

#### 6. 문제 푼 시간에 대한 PCA
def feat_elapsed_pca(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = feat_elapsed(df)
    df = feat_elapsed_cumsum(df)
    df = feat_normalized_elapsed(df)
    df = feat_elapsed_type_stats(df, 'mean')
    df = feat_relative_elapsed_time(df)

    X = df[['elapsed', 'elapsed_cumsum', 'normalized_elapsed', 'relative_elapsed']]

    pca = PCA(n_components=1)
    df['elapsed_pca'] = pca.fit_transform(X)
    
    return df

#### 7. 문제 푼 시간에 대한 LDA
def feat_elapsed_lda(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = feat_elapsed(df)
    df = feat_elapsed_cumsum(df)
    df = feat_normalized_elapsed(df)
    df = feat_elapsed_type_stats(df, 'mean')
    df = feat_relative_elapsed_time(df)

    X = df[['elapsed', 'elapsed_cumsum', 'normalized_elapsed', 'relative_elapsed']]
    y = df['answerCode']

    lda = LDA(n_components=1)
    df['elapsed_lda'] = lda.fit_transform(X, y)

    return df

## 이동 평균 (Rolling Mean) Reature 만들기
#### 1. 최근 n개 문제 평균 풀이 시간
def feat_rolling_mean_time(df : pd.DataFrame, n : int) -> pd.DataFrame:
    df = df.copy()
    # Timestamp -> 초단위 변환
    df['elapsed'] = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    df['elapsed'] = df['elapsed'].apply(lambda x: x.total_seconds()).shift(-1, fill_value=0)
    
    # 문제 푸는데 10분이상 걸리면 0으로 초기화
    df['elapsed'] = df['elapsed'].apply(lambda x: 0 if x >= 600 else x)
    
    # 최근 n개 문제 평균 풀이 시간
    df['rolling_mean_time'] = df.groupby(['userID'])['elapsed'].rolling(n).mean().round(3).values
    df['rolling_mean_time'] = df['rolling_mean_time'].fillna(0)
    
    return df

def dataloader_group_values(r):
    columns = list(r.columns)
    columns.remove('userID')
    list_temp = []
    for col in columns:
        list_temp.append(r[col].values)
    
    return tuple(list_temp)