import pickle
import time
import torch
from neural_pde.utils.parser import get_training_parser
from neural_pde.nn.nn_fns import get_model
from neural_pde.data.data_fns import get_loaders
from neural_pde.loss.loss_fns import get_loss
from neural_pde.opt.opt_fns import get_opt

import wandb


LOG_MOD = 20

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_proj)
        wandb.config.update(args.__dict__)
    
    if args.prop_log:
        PROP_LOG = args.LOG_MOD
    
    loaders = get_loaders(args)

    model = get_model(args)
    opt = get_opt(args)
    loss_fn = get_loss(args)

    tic = time.time()
    # train
    model.to(device)
    model.train()
    print(f"Model Type: {args.model}")
    print(f"Training Set Number of Batches: {len(train_loader)}")
    print(f"Total iterations : {args.iters}")

    i = 0
    for idx, batch in zip(range(args.iters), train_loader):
        x, y = batch[0].to(device), batch[1].to(device)
        y_hat = model(x)
        loss = loss_fn(y, y_hat)
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Log other metrics here

        if args.wandb and idx % LOG_MOD == 0:
            wandb.log({
                "loss": loss.item(),
                "iter": i,
                # add other metrics here
            })
        
        # Print other metrics here
    
    # save model
    toc = time.time()
    wandb.log({"training_time": toc - tic})




if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    main(args)
