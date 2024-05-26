import os
import argparse

import torch


# Add args: loss_fn [cross_entropy, dice]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        required=True,
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help=
        "Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
        " Note that same models used for all datasets, so much have same classnames"
        "for zero shot.",
    )
    parser.add_argument(
        "--final-eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help=
        "Which datasets to use for evaluation at the end of the training. Split by comma, e.g. CIFAR101,CIFAR102."
        " Note that same models used for all datasets, so much have same classnames"
        "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )

    parser.add_argument(
        "--loss-fn",
        default=None,
        type=lambda x: x.split(","),
        help="Which loss function to use. It can be a list of loss functions, e.g. cross_entropy,dice",
    )

    parser.add_argument(
        "--loss-weights",
        default=None,
        type=lambda x: x.split(","),
        help="Each one of the weights to apply to the loss fns",
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.")
    parser.add_argument(
        "--model",
        type=str,
        # choices=['vit_b', 'vit_h', 'vit_l', 'vit_b_fourier', 'vit_b_domain', 'vit_b_adapter', 'vit_b_encoder_adapter',
        #          'vit_b_biggeradapter'],
        default='vit_b',
        help="The type of models (e.g. vit_b, vit_h...).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None
    )
    parser.add_argument(
        "--pretrained-decoder",
        type=str,
        default=None
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["constant", "cosine", "exponential"],
        help="Which scheduler to use",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "custom"],
        help="What optimizer to use",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0,
        help="Momentum for SGD",
    )
    parser.add_argument(
        "--betas",
        default=[0.9, 0.999],
        nargs='+',
        type=float,
        help="Betas for AdamW",
    )

    parser.add_argument("--workers",
                        type=int,
                        default=8,
                        help="Number of dataloader workers per GPU.")

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine scheduler",
    )

    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="Use wandb",
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='',
        help='If left blank, will generate a project name based on other arguments'
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
    )

    parser.add_argument(
        '--task',
        choices=['training', 'testing', 'training_iterative', 'instance_testing'],
        type=str,
        default='training'
    )

    parser.add_argument(
        '--lora-ckpt',
        type=str,
        default=None
    )

    parser.add_argument(
        '--training-split',
        choices=['train', 'val', 'test', 'trainval'],
        type=str,
        default='train',
    )

    parser.add_argument(
        '--baseline',
        type=str,
        choices=['medsa', 'lora', 'decfinetune', 'finetune', ''],
        default=''
    )

    parser.add_argument(
        '--test-split',
        type=str,
        choices=['train', 'val', 'test'],
        default='val',
    )

    parser.add_argument(
        '--testing-ckpt',
        type=str,
        default=None
    )

    parser.add_argument(
        '--write-freq',
        type=int,
        default=10,
    )

    parser.add_argument(
        "--prompting",
        default=False,
        action="store_true",
        help="Use bbox prompting",
    )

    parser.add_argument(
        "--dataset-type",
        default='all',
        choices=['all', 'old', 'new'],
        help="Which dataset to use",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
