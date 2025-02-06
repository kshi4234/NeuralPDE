"""
Get different neural network models
- MLP
- Transformer
"""

from neural_pde.transformers.base_transformer import Transformer, TransformerConfig


def get_model(args):
    if args.model == "transformer":
        config = TransformerConfig(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            bias=args.bias,
            dropout=args.dropout    
        )
        return Transformer(config)
    else:
        raise ValueError(f"Model {args.model} not supported")
