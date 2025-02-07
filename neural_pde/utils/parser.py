import argparse

def get_training_parser():
    parser = argparse.ArgumentParser(description="GP-PDE")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--datapath", type=str, default="./datasets")
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_proj", type=str, default="gp-pde")
    parser.add_argument("--operation", type=str, default="largest_eigenvalue")
    parser.add_argument("--distribution", type=str, default="normal")
    parser.add_argument("--description", type=str, default="")
    return parser

def get_data_parser():
    parser = argparse.ArgumentParser(description="GP-PDE Data")
    parser.add_argument("--shape", type=str, default="square")
    return parser