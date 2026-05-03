import logging
import os
import wandb
import torch.utils.tensorboard as tensorboard
import torch
from torch.cuda.amp import GradScaler
from timm.utils import ModelEma
from model.model import get_model
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging, get_exp_name
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch
from training.optimization import get_optimizer
from evaluation.evaluation_test import evaluate_test
from data.train_data import get_data
from data.episodic_training import init_index_mapping, update_index_mapping
from loss import get_loss

def main():
    args = parse_args()
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.name = get_exp_name(args)
    args.log_path = None

    # Set logger
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    init_distributed_device(args)

    # init wandb & tensorboard logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    model, preprocess_train, preprocess_val, preprocess_aug = get_model(args)
    save_path=args.resume
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    evaluate_test(model,preprocess_val, args,writer)


if __name__ == "__main__":
    main()

