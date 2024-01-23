import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="gpu", type=str, help="cpu or gpu")
    
    parser.add_argument("--data_dir", default="/data/ephemeral/data", type=str, help="data directory")
    parser.add_argument("--asset_dir", default="asset/", type=str, help="data directory")
    parser.add_argument("--model_dir", default="models/", type=str, help="model directory")
    parser.add_argument("--output_dir", default="outputs/", type=str, help="output directory")
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="model file name")
    parser.add_argument("--file_name", default="train_data_feature.csv", type=str, help="train file name")
    # parser.add_argument("--file_name", default="train_data_merge.csv", type=str, help="train file name")
    parser.add_argument("--test_file_name", default="test_data.csv", type=str, help="test file name")
    
    #### 모델 선언 ####
    parser.add_argument("--model", default="XGBCBoost", choices=['AdaBoost', 'GradBoost, XGBCBoost, CatBoost', 'LGBM'], type=str, help="model select")
    
    # Boost 모델 공통
    parser.add_argument('--test_size', default=0.2, type=float, help='Train/Valid split 비율을 조정할 수 있습니다.')
    parser.add_argument("--n_estimators", default=500, type=int, help="number of epochs")
    parser.add_argument('--data_shuffle', default=True, type=bool, help='데이터 셔플 여부를 조정할 수 있습니다.')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    
    parser.add_argument('--kfold', default='', choices=['', 'kfold'], type=str, help='choice kfold')
    parser.add_argument('--n_fold', default=5, type=int, help='Number of Kfold')
    
    # if parser.parse_args().model == 'AdaBoost':
    #     pass
    # elif parser.parse_args().model == 'GradBoost':
    #     parser.add_argument('--subsample', default=1.0, type=float, help='')
    #     parser.add_argument('--min_samples_split', default=2, type=float, help='')
    #     parser.add_argument('--min_samples_leaf', default=1, type=float, help='')
    #     parser.add_argument('--min_weight_fraction_leaf', default=0.0, type=float, help='')
    #     parser.add_argument('--max_depth', default=3, type=int, help='')
    #     parser.add_argument('--min_impurity_decrease', default=0.0, type=float, help='')
    #     parser.add_argument('--verbose', default=0, type=int, help='')
    #     parser.add_argument('--validation_fraction', default=0.1, type=float, help='')
    #     parser.add_argument('--tol', default=0.0001, type=float, help='')
    #     parser.add_argument('--ccp_alpha', default=0.0, type=float, help='')
    # elif parser.parse_args().model == 'XGBCBoost':
    #     parser.add_argument('--max_depth', default=8, type=int, help='')
    #     parser.add_argument('--colsample_bylevel', default=0.9, type=float, help='')
    #     parser.add_argument('--colsample_bytree', default=0.8, type=float, help='')
    #     parser.add_argument('--gamma', default=0, type=int, help='')
    #     parser.add_argument('--min_child_weight', default=3, type=int, help='')
    #     parser.add_argument('--nthread', default=4, type=int, help='')
    # elif parser.parse_args().model == 'CatBoost':
    #     pass
    # elif parser.parse_args().model == 'LGBM':
    parser.add_argument('--max_depth', default=-1, type=int, help='Tree의 최대 깊이 -1이 효과적')
    parser.add_argument('--min_data_in_leaf', default=20, type=int, help='Leaf')
    parser.add_argument('--feature_fraction', default=0.8, type=float, help='Boosting이 랜덤 포레스트일 경우 사용합니다')
    parser.add_argument('--_lambda', default=0.0, type=float, help='regularization 정규화를 합니다. : 0 ~ 1')
        
    parser.add_argument("--log_steps", default=50, type=int, help="print log per n steps")

    args = parser.parse_args()

    return args
