# Baseline3: Boosting
## Package Version
```bash
numpy==1.26.3
pandas==2.1.4
ipykernel==6.29.0
scikit-learn==1.3.2
lightgbm==4.1.0
catboost==1.2.2
xgboost==2.0.3
wandb==0.16.2
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
![Boosting_Flow_Chart](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/8871767/4031ba71-8ec2-4232-ab36-8fbc3e55f7bc)
## Files Tree
```
📦boosting
┣ 📂boosting
┃ ┣ args.py
┃ ┣ dataloader.py
┃ ┣ metric.py
┃ ┣ model.py
┃ ┣ trainer.py
┃ ┗ utils.py
┣ 📂models
┣ 📂outputs
┣ main.py
┣ README.md
┣ requirements.txt
┃ wandb_sweep.yaml
```

`📦boosting`
* `main.py` : main() 함수로 최초 실행하는 파일입니다.
* `requirements.txt` : 패키지 실행에 필요한 라이브러리들이 정리되어 있습니다.
* `wandb_sweep.yaml` : wandb(Weights and Biases) 툴의 Sweep을 실행하기 위한 파일입니다.

`📦boosting/📂boosting`
* `args.py` : 학습에 활용되는 여러 argument가 선언되어 있습니다.
* `dataloader.py` : 학습에 필요한 파일을 불러오고 Featrue Engineering와 Pre-processing을 진행합니다.
* `metric.py` : metric이 정의되어 있는 함수를 포함합니다.
* `model.py` : Boosting계열 모델을 정의하는 클래스를 포함합니다.
* `trainer.py` : Model Train과 관련된 함수를 포함합니다.
* `utils.py` : 학습에 필요한 부수적인 함수들을 포함합니다.

`📦boosting/📂model`
* K-fold를 진행할 때 생성되는 모델을 저장하는 폴더입니다.

`📦boosting/📂outputs`
* 학습이 완료된 모델에서부터 나온 Inference 결과를 저장하는 폴더입니다.
