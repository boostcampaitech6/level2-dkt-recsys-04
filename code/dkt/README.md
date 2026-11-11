# Baseline1: Deep Knowledge Tracing
## Package Version
```bash
torch
pandas
scikit-learn
tqdm
wandb
transformers
```
## Setup
```bash
cd /opt/ml/input/code/lightgcn
conda init
(base) . ~/.bashrc
(base) conda create -n gcn python=3.10 -y
(base) conda activate gcn
(gcn) pip install -r requirements.txt
(gcn) python train.py
(gcn) python inference.py
```
## Flow Chart
![model_seq](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/95879995/c7e66caf-d0fc-462c-9202-b4a9b02893fb)
)
## Files Tree
```
📦 dkt
├─📂 asset
│  ├─ KnowledgeTag_classes.npy
│  ├─ assessmentItemID_classes.npy
│  ├─ paper_number_classes.npy
│  └─ testId_classes.npy
├─📂 dkt
│  ├─ __pycache__
│  ├─📂 feature_engineering
│  ├─ args.py
│  ├─ config.yaml
│  ├─ criterion.py
│  ├─ dataloader.py
│  ├─ metric.py
│  ├─ model.py
│  ├─ optimizer.py
│  ├─ scheduler.py
│  ├─ trainer.py
│  └─ utils.py
├─ 📂wandb
├─ README.md
├─ inference.py
├─ requirements.txt
└─ train.py
```

`📦dkt`
* `train.py` : main() 함수로 최초 실행하는 파일입니다.
* `inference.py` : submission 제출을 위해 결과값을 도출 하는 파일입니다.
* `requirements.txt` : 패키지 실행에 필요한 라이브러리들이 정리되어 있습니다.
* `wandb_sweep.yaml` : wandb(Weights and Biases) 툴의 Sweep을 실행하기 위한 파일입니다.

`📦dkt/📂asset`
* `KnowledgeTag_classes.npy` : KnowledgeTag의 encoder data
* `assessmentItemID_classes.npy` : assessmentItemID_classes의 encoder data
* `paper_number_classes.npy` : paper_number_classes의 encoder data
* `testId_classes.npy` : testId_classes의 encoder data


`📦dkt/📂dkt`
* `args.py` : 학습에 활용되는 여러 argument가 선언되어 있습니다.
* `config.yaml` : sweep을 돌리기위한 설정 파일입니다.
* `criterion.py` : loss 계산을 위한 파일입니다.
* `dataloader.py` : 학습에 필요한 파일을 불러오고 Featrue Engineering와 Pre-processing을 진행합니다.
* `metric.py` : metric이 정의되어 있는 함수를 포함합니다.
* `model.py` : Boosting계열 모델을 정의하는 클래스를 포함합니다.
* `optimizer.py` : optimizer 설정을 위한 파일입니다.
* `scheduler.py` : scheduler 설정을 위한 파일입니다.
* `trainer.py` : Model Train과 관련된 함수를 포함합니다.
* `utils.py` : 학습에 필요한 부수적인 함수들을 포함합니다.


`📦dkt/📂wandb`
* wandb 학습 결과를 저장하기 위한 폴더입니다.







