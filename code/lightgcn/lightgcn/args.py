import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", default=True, type=bool, help="Use GPU")
    
    parser.add_argument("--data_dir", default="/data/ephemeral/data/", type=str, help="")
    
    parser.add_argument("--output_dir", default="./outputs/", type=str, help="")
    
    parser.add_argument("--hidden_dim", default=64, type=int, help="")
    parser.add_argument("--n_layers", default=1, type=int, help="")
    parser.add_argument("--alpha", default=None, type=float, help="")
    
    parser.add_argument("--n_epochs", default=20, type=int, help="")
    parser.add_argument("--lr", default=0.001, type=float, help="")
    parser.add_argument("--model_dir", default="./models/", type=str, help="")
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="")
    
    # feat siyun
    ## add argument --purpose
    parser.add_argument('--purpose', choices = ['result','embedding'] , default = 'result' ,type=str, help = "lightgcn의 결과를 어떻게 사용할지 정합니다.")

    args = parser.parse_args()

    return args
