from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mlflow
import datetime

from src.pytorch.models import Net
from src.pytorch.dataloader import dataloader
from src.pytorch.plot import plot_learning
from src.pytorch.learning import train, valid, learning


def make_parse():
    """
    コマンドライン引数を受け取る
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--exp_name', default=str(datetime.datetime.now()), type=str, help='experiment_name')
    return parser


def main():
    # Training settings
    args = make_parse().parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # MLflow 保存先/実験名
    tracking_uri = "../../mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = args.exp_name
    mlflow.set_experiment(experiment_name)

    # data_loader
    train_loader, test_loader = dataloader(args, device)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 学習、精度評価
    train_loss_list, valid_loss_list = learning(
        args=args,
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        # criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # 学習曲線
    plot_learning(train_loss_list, valid_loss_list, '../../data/output/loss.png')

    # mlflow
    with mlflow.start_run() as run:  # runIDを発行する

        # 時間で経過する数値を記録する(loss等)
        for idx, loss in enumerate(train_loss_list, 1):
            mlflow.log_metric("train_loss", loss, idx)

        for idx, loss in enumerate(valid_loss_list, 1):
            mlflow.log_metric("valid_loss", loss, idx)

        # pngファイルを記録
        mlflow.log_artifact(local_path='../../data/output/loss.png')

        # パラメータを記録 (key-value pair)
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # mlflow.pytorch.log_model(model, 'model')  # PyTorchモデルを現在実行中のMLflowアーティファクトとしてログに記録


if __name__ == '__main__':
    main()
