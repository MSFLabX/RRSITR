import logging
import os
import json
import torch
from training.distributed import is_master
from .linear_eval import linear_eval
from .zero_shot import zero_shot_eval
from .retrieval import retrieval_evaluation
# from .analyze_features import analyze_features
# from .sts_evaluation import sts_benchmark
from .nlp_evaluations import nlp_eval
from .wise_ft import get_wise_ft_model

try:
    import wandb
except ImportError:
    wandb = None


def evaluate_test(model,preprocess, args, tb_writer=None):
    if args.distributed and not is_master(args):
        return
    logging.info(f"Starting evaluation of [{args.name}]")

    if args.eval_with_wise_ft != 1:
        logging.info(f"Perform Wise-FT evaluation with alpha={args.eval_with_wise_ft}")
        model = get_wise_ft_model(model, args, alpha=args.eval_with_wise_ft)
        distributed = args.distributed
        args.distributed = False

    if args.model_ema:
        distributed = args.distributed
        args.distributed = False

    linear_eval_datasets = ['CIFAR10']
    zeroshot_datasets = ['ImageNet']
    args.evaluation_workers = 8

    model.eval()
    all_metrics1 = {}
    save_best_val = {'best_mean_recall': 0.0, 'best_epoch': -1}
    if not hasattr(args, 'best_retrieval_metrics'):
        args.best_retrieval_metrics = {}

    # Image-text retrieval
    args.retrieval_data = args.retrieval_data1
    retrieval_metrics1 = retrieval_evaluation(model, preprocess, args)
    all_metrics1.update(retrieval_metrics1)
    logging.info(f"Finished evaluation1 of [{args.name}] \n" + "\n".join(
        [f"\t{k}\t{v}" for k, v in all_metrics1.items()]))
    current_data = args.retrieval_data
    current_mean_recall = retrieval_metrics1.get("retrieval-mean-recall", 0.0)
    best_entry = args.best_retrieval_metrics.get(current_data, {'best_mean_recall': 0.0, 'best_epoch': 0})
    if current_mean_recall > best_entry['best_mean_recall']:
        best_entry.update({
            'best_mean_recall': current_mean_recall,
        })
        args.best_retrieval_metrics[current_data] = best_entry
        save_best_val = best_entry

    if args.save_logs:
        with open(os.path.join(args.logs, args.name, "results.jsonl"), "a+") as f:
            f.write(json.dumps(all_metrics1))
            f.write("\n")

        best_metrics_path = os.path.join(args.logs, args.name, "best_retrieval_metrics.json")
        with open(best_metrics_path, "w") as f:
            json.dump(args.best_retrieval_metrics, f, indent=4)

    if args.eval_with_wise_ft != 1 or args.model_ema:
        args.distributed = distributed

    return save_best_val
