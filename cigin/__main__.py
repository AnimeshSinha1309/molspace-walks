import warnings
import argparse

from rdkit import RDLogger
from rdkit import rdBase

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

from cigin.model import CIGINModel
from cigin.train import train
from cigin.dataset import csv_to_loader


RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")


def main(interaction, max_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = csv_to_loader('solvatum/data/train.csv', batch_size=32, shuffle=True)
    valid_loader = csv_to_loader('solvatum/data/valid.csv', batch_size=128)

    model = CIGINModel(interaction=interaction)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)

    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', help="interaction function: dot | scaled-dot | general | tanh-general", default='dot')
    parser.add_argument('--epochs', type=int, default=100, help="The max number of epochs for training")
    args = parser.parse_args()
    main(args.fn, int(args.epochs))
