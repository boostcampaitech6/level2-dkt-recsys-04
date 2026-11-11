lightgcn의 모델링 결과를 받아 mf로 처리하고자 합니다.

일단 참고는 https://github.com/boostcampaitech5/level1_bookratingprediction-recsys-03/blob/main/chanwoong/models/FFMDCN.py

추가적으로 더 참고할 수도 있음


how to run
1. train.py --purpose 'result' or 'embedding'
2. main.py --purpose 'result'

args.py
- --purpose 추가
- --data_dir : 본인의 경로에 맞게 수정필요


main.py
- 01.17 - 현재 수정 중.
- lightgcn의 최종 결과인 users, 예측된 answercode를 인풋으로 사용하고, 추가적인 feature와 같이 MF를 진행합니다.



train.py
- lightgcn의 학습이 일어납니다.
- purpose는 ['result','embedding']으로 나누어집니다.
    - result : lightgcn의 최종 학습 결과를 받습니다.
    - embedding : lightgcn의 학습 결과에 대한 user,item에 대한 embedding으로 반환합니다.
        - how to open
            ```
            path = args.data_dir
            model = torch.load(path + 'embeddings.pt')
            ```


datasets.py
- 추가적인 feature를 추가하고자 하였으나, 성능 + 라이브러리 문제로 주석처리 되었습니다.

trainer.py
- run : purpose에 따라 다르게 처리합니다.



