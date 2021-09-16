from pytorch_pretrained_bert import BertAdam
import torch.optim as optim

def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt", "mmbt_feat"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
        )
        # optimizer = optim.Adadelta(model.parameters(),lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    return optimizer