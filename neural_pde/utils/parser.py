import argparse

def get_training_parser():
    parser = argparse.ArgumentParser(description="PDE")
    # generic args for all models
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)

    # transformer args
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)

    # neural network args
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)

    # gaussian process args
    parser.add_argument("--gp_mean", type=str, default="zero")
    parser.add_argument("--gp_covar", type=str, default="rbf")
    parser.add_argument("--gp_likelihood", type=str, default="gaussian")

    # deep kernel gp args
    parser.add_argument("--feature_extractor_hidden_layers", type=int, default=3)
    parser.add_argument("--feature_extractor_hidden_dim", type=int, default=128)
    parser.add_argument("--feature_extractor_input_dim", type=int, default=1)
    parser.add_argument("--feature_extractor_output_dim", type=int, default=1)

    # data args
    parser.add_argument("--datapath", type=str, default="./datasets")
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_proj", type=str, default="gp-pde")
    parser.add_argument("--operation", type=str, default="largest_eigenvalue")
    parser.add_argument("--distribution", type=str, default="normal")
    parser.add_argument("--description", type=str, default="")

    return parser

def get_data_parser():
    parser = argparse.ArgumentParser(description="PDE Data")
    parser.add_argument("--shape", type=str, default="square")
    return parser