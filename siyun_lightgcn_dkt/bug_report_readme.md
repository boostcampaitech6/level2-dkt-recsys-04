 lightgcn의 결과를 dkt에 옮기는 과정입니다.

최종으로 예측되는 output결과는 다음 그림과 같습니다.

![lightgcn_to_other](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/68991530/f7197e98-9c0a-4caf-9eda-642cd1597b07)

 사용방법
 - python train.py
    - 임베딩을 받아서 train을 처리합니다.

- python inference.py --purpose 'inference'
    - train.py, inference.py를 제외한 나머지를 공유하기 때문에 다음과 같이 정의하였습니다.
    - inference로 설정시켜야 합니다.


현 문제점
- train.py의 구조를 inference에서 똑같이 만들었습니다.

train의 구조
- load_train_data에서 train data를 불러옵니다.
```
preprocess.load_train_data(file_name=args.file_name)
```
- 여기에서 load_data_from_file을 통해 user2index, item2index를 반환합니다.
```
user2index = preprocess.user2index
item2index = preprocess.item2index
```
- 이를 바탕으로 get_model에 user2index, item2index를 넣습니다.
```
37줄
model = trainer.get_model(args, user2index, item2index).to(args.device)
```
- 넣어진 user2index, item2index는 lstm에 들어가는데, lstm은 modelbase를 상속받기 때문에 여기에서 train에 대한 임베딩을 저장된 경로를 통해 불러옵니다.
```
self.user2index = user2index
        self.item2index = item2index
        print("Loading embeddings...")
        # note siyun : uset_emb, item_Emb 위치 변경해서 다시 try
        if args.purpose == 'train' :
            self.user_emb = torch.load('/data/ephemeral/siyun_lightgcn_2_fm/models/user_emb.pt')
            self.item_emb = torch.load('/data/ephemeral/siyun_lightgcn_2_fm/models/item_emb.pt')
            print('train embedding shape is : ')
            print(self.user_emb.shape)
            print(self.item_emb.shape)
        
        elif args.purpose == 'inference' :
            self.user_emb = torch.load('/data/ephemeral/siyun_lightgcn_2_fm/models/test_user_emb.pt')
            self.item_emb = torch.load('/data/ephemeral/siyun_lightgcn_2_fm/models/test_item_emb.pt')
            print('test embedding shape is : ')
            print(self.user_emb.shape)
            print(self.item_emb.shape)
            ```

- 이제 train에 먼저 넣어진 feature들을 임베딩화 시킨 후 나중에 lightgcn의 임베딩과 결합합니다.
```
```
model.py line 77
total_embedding_dim = intd * (len(self.args.cat_cols) + 1) + len(self.args.con_cols) + user_emb_dim + item_emb_dim
~~~
line 92
self.comb_proj = nn.Linear(total_embedding_dim, hd)
```

- 불러온 임베딩은 dktdataset에서 패딩부분이 적용된 인덱스와 결합되어 각 index에 해당하는 좌표로 결합됩니다.
```forward
line 106~107
user_indices = data['user_idx']
item_indices = data['item_idx']
        ```
- 임베딩의 차원을 맞춰서 오류가 나지 않도록 조절합니다.
``` line131~132
user_emb_batch = self.user_emb[user_indices].view(batch_size, seq_len, -1)
item_emb_batch = self.item_emb[item_indices].view(batch_size, seq_len, -1)
```

inference의 구조
- load_test_data에서 test data를 불러옵니다.
```inference.py line 23
preprocess.load_test_data(file_name=args.test_file_name)
```
- 여기에서 load_data_from_file을 통해 test의 구조에 맞춘 user2index, item2index를 반환합니다.
```line 26~27
user2index = preprocess.user2index
item2index = preprocess.item2index
```

- 이를 바탕으로 load_model에 넣습니다.
```line 38
model = trainer.load_model(args, trainer.get_model, user2index, item2index).to(args.device)
```

load_model은 안에 get_model을 갖는 구조로, 인자에 user2index, item2index를 넣어 get_model에 넣게 합니다.
```
trainer.py load_model line 337
model = get_model(args,user2index, item2index)
```
 
- get_model에 들어가서 학습을 진행하고 결과를 inference로 받습니다.
```
trainer.py line 270~
```

Make Predictions & Save Submission ...여기까지는 진행됩니다.

```
inference.py line 41
logger.info("Make Predictions & Save Submission ...")
```


- trainer.py의 inference에 들어가 get_loaders의 결과를 받습니다.
```trainer.py line 240~
_, test_loader = get_loaders(args=args, train=None, valid=test_data)
```
- trainer.py의 249줄의 preds = model(batch)를 실행합니다.
    - lstm의 forward를 진행합니다.
    ```
    def forward(self, data):
        # X는 embedding들을 concat한 값
        # super().forward은 부모객체의 forward메소드를 말함
        # X, batch_size, _ = super().forward(data) # [건우] 각각 안 받고 data로 한 번에 받음
        enc_X, X, batch_size, seq_len = super().forward(data)
    ```
    - 이 부분을 진행하게 됩니다.
    - forward를 진행하기 위해 modelbase에 들어가는데 이때
    ```
    File "/data/ephemeral/code/dkt/dkt/model.py", line 126, in forward
    raise ValueError("User index out of range in user_emb.")
    ValueError: User index out of range in user_emb.
    ```
    이 오류가 나옵니다.

오류를 확인해본 결과 test의 임베딩은 744개 정도로 나오는데, dktdataset에서 그 이상의 index를 반환하는 것으로 나타났습니다.
