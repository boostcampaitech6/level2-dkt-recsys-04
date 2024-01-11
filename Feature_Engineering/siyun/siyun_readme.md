# fe_siyun.py

1. elapsed
- timestamp를 바탕으로 문제를 푸는 데 소요한 시간에 따른 정답률

2. cumsum
- 문제를 풀었을 때 누적량에 따른 정답률

3. type
    1. test_type
    - 모두 A로 고정됨
    2. paper_type
    - 0~9로 시험지 대분류로 가정
    3. paper_subtype
    - 시험지 중분류로 가정

4. type_percent
- 위에서 처리한 type을 변환하여 각각의 정답률 처리

5. check_components
- kmeans로 몇개의 그룹이 최적화 대상이 될 수 있는지 확인
- 여기서는 태그를 기준으로 묶음

6. svd
- TruncatedSVD를 진행
- 단일 칼럼에 대한 차원축소를 진행하기 위해 차원을 늘린후, 다시 줄이는 방식인 SVD를 차용
- 여러 칼럼을 바탕으로 차원 축소를 진행하기 위해서 PCA도 경우에 따라서는 사용해도 좋음.
- 진행 결과는 KnowledgeTag_SVD_n 으로 각각의 주성분에 따라서 칼럼 생성하여 사용

7. total_input
- svd를 사용한 결과를 포함하여 최종적으로 모델의 input에 넣기위한 feature를 반환