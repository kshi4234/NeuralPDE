import pickle
import time
import torch
from neural_pde.utils.parser import get_training_parser
from neural_pde.models.model import get_model
from neural_pde.loss.losses import get_loss

import wandb


LOG_MOD = 20

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.wandb:
        wandb.init(project=args.wandb_proj)
        wandb.config.update(args.__dict__)
    
    

    model = get_model(args)

    exit()

    tic = time.time()
    # train
    model.to(device)
    model.train()
    print(f"Model Type: {args.model}")

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
            })
        
        # Print other metrics here
    
    # save model
    toc = time.time()
    wandb.log({"training_time": toc - tic})




if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    main(args)
