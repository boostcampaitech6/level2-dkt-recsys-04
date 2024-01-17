import argparse


def parse_args():
    # python script 실행할 때 인자를 수정하여 실행할 수 있다.
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="cpu or gpu") 
    parser.add_argument(
        "--data_dir",
        default="/opt/ml/input/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="outputs/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=300, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # [건우] feature 분류모음(자동화 위해 추가)
    parser.add_argument(
        "--cat_cols",
        default=["testId", "assessmentItemID", "KnowledgeTag"], # "userID"로 묶을 것이기 때문에 "userID"는 제외
        type=list,
        help="categorical features",
    )
    parser.add_argument(
        "--con_cols",
        default=["elapsed", "KnowledgeTag_percent", "cumulative","paper_number"],
        type=list,
        help="numerical features",
    )
    parser.add_argument(
        "--tgt_col", default=["answerCode"], type=list, help="target feature"
    )
    

    # 모델
    parser.add_argument(
        "--hidden_dim", default=128, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=1, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=4, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.0, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=2.0, type=int, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="saint", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )

    # argparse.ArgumentParser()객체 안에 parse_args()라는 메소드가 있음 -> def parse_args()의 parse_args()가 아님
    args = parser.parse_args() # ex) Namespace(...,model='lstm', optimizer='adam', ...)

    return args # python script 실행할 때 인자를 수정하여 실행할 수 있다.
