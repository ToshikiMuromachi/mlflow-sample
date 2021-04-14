import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import datetime
from dataset import generate_loader
from models import Net
from utils import train

def make_parse():
    """
    コマンドライン引数を受け取る
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning_rate')
    parser.add_argument('--exp_name', default=datetime.datetime.now(), type=str, help='experiment_name')
    return parser


def main():
    """
    実行関数
    """
    #コマンドライン引数
    args = make_parse().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MLflow 保存先/実験名
    tracking_uri = "../../mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = args.exp_name
    mlflow.set_experiment(experiment_name)

    # data_loader
    train_loader, test_loader = generate_loader(args.batch_size)

    # model
    model = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs*0.8), int(args.epochs*0.9)],
        gamma=0.1
    )

    model = model.to(device)

    # 学習、精度評価 
    results = train(
        epochs=args.epochs, 
        model=model, 
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # mlflow
    with mlflow.start_run() as run:    # runIDを発行する

        for key, value in results.items():    # メトリックを記録する(ステップごとに)
            mlflow.log_metric(key, value)

        for key, value in vars(args).items():    # パラメータを記録する (key-value pair)
            mlflow.log_param(key, value)

        mlflow.pytorch.log_model(model, 'model')     # 生成物を記録する (output file)

if __name__ == '__main__':
    main()