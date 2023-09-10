# AI Model based on a graph (GIN) for the Emotion Recognition
#
# @author Emilio Garzia
#
# i : Run tensorboard to see all progress of the training/test phase.
#     Open another terminal and insert this command: python3 -m tensorboard.main --logdir=train/loss 

from typing import List, Tuple
import torch.nn as nn
from torcheeg import transforms
from torcheeg.transforms.pyg import ToG
from torcheeg.datasets import SEEDIVFeatureDataset, SEEDIVDataset
from torcheeg.datasets.constants.emotion_recognition.seed_iv import SEED_IV_ADJACENCY_MATRIX, SEED_IV_ADJACENCY_LIST, SEED_IV_CHANNEL_LIST
from torcheeg.datasets.constants.utils import format_adj_matrix_from_adj_list
from torcheeg.models.pyg import GIN 
from torcheeg.trainers import ClassificationTrainer
from torch_geometric.loader import DataLoader
from torcheeg.model_selection import KFoldGroupbyTrial
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import numpy as np
from rich import print
from rich.progress import track
import argparse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 25

#################### Classification Trainer ######################
class MyClassificationTrainer(ClassificationTrainer):
    def __init__(self, model: nn.Module, num_classes=None, lr: float = 0.0001, weight_decay: float = 0, device_ids: List[int] = [], ddp_sync_bn: bool = True, ddp_replace_sampler: bool = True, ddp_val: bool = True, ddp_test: bool = True):
        super().__init__(model, num_classes, lr, weight_decay, device_ids, ddp_sync_bn, ddp_replace_sampler, ddp_val, ddp_test)
        self.writer = SummaryWriter("train/loss")
        self.steps_file_name = "train/steps"
        self.train_counter, self.test_counter = self.load_counters()
        self.last_train_loss = 0
        self.last_train_accuracy = 0

    def on_training_step(self, train_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        super().on_training_step(train_batch, batch_id, num_batches, **kwargs)
        if self.train_loss.mean_value.item() != 0:
            self.last_train_loss = self.train_loss.compute()
            self.last_train_accuracy = self.train_accuracy.compute()

    def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
        super().after_validation_epoch(epoch_id, num_epochs, **kwargs)
        torch.save(model.state_dict(), "./train/model.pth")
        self.writer.add_scalars('loss', {
            'train': self.last_train_loss,
            'validation': self.val_loss.compute()
        }, self.train_counter)
        self.writer.add_scalars('accuracy', {
            'train': self.last_train_accuracy*100,
            'validation': self.val_accuracy.compute()*100
        }, self.train_counter)
        self.train_counter += 1
        self.save_counters()

    def after_test_epoch(self, **kwargs):
        super().after_test_epoch(**kwargs)
        self.writer.add_scalar("loss/test", self.test_loss.compute(), self.test_counter)
        self.writer.add_scalar("accuracy/test", self.test_accuracy.compute()*100, self.test_counter)
        self.test_counter += 1
        self.save_counters()

    def load_counters(self):
        if os.path.exists(self.steps_file_name):
            step_file = open(self.steps_file_name, "r")
            values = step_file.readline().split(",")
            return int(values[0]), int(values[1])
        return 0, 0

    def save_counters(self):
        steps_file = open(self.steps_file_name, "w")
        steps_file.write(f"{self.train_counter},{self.test_counter}")
        steps_file.flush()
        steps_file.close()


if __name__ == '__main__':
    #################### 16 CHANNELS EXTRACTION ######################
    """UMIV_CHANNEL_LIST = ['FP1', 'FP2', 'F7', 'F3', 'F4',
                         'F8', 'T7', 'C3', 'C4', 'T8', 'P7',
                         'P3', 'P4', 'P8', 'O1', 'O2']
    
    SEED_IV_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(UMIV_CHANNEL_LIST, SEED_IV_ADJACENCY_LIST)
    """

    #################### Load Dataset ######################
    """ # Raw Dataset
    dataset = SEEDIVDataset(io_path=f'./dataset/seed_iv',
                      root_path='./dataset/eeg_raw_data',
                      online_transform=transforms.Compose([
                          ToG(SEED_IV_ADJACENCY_MATRIX)
                      ]),
                      label_transform=transforms.Select('emotion'))
    """

    # Features Dataset
    dataset = SEEDIVFeatureDataset(io_path=f'./dataset/seed_iv',
                      root_path='./dataset/eeg_feature_smooth',
                      feature=["psd_movingAve", "de_movingAve"],
                      online_transform=transforms.Compose([transforms.MinMaxNormalize(),ToG(SEED_IV_ADJACENCY_MATRIX)]),
                      label_transform=transforms.Select('emotion'))


    #################### Init AI Model [GNN] ######################
    model = GIN(in_channels=10, hid_channels=64, num_classes=4).to(device=device)

    #################### Checkpoint Training File ######################
    try:
        model.load_state_dict(torch.load("./train/model.pth"))
    except FileNotFoundError:
        print("Primo addestramento!")

    #################### Init Trainer ######################
    trainer = MyClassificationTrainer(model=model, lr=1e-4, weight_decay=1e-4)
    k_fold = KFoldGroupbyTrial(n_splits=5, split_path='./dataset/split/')
    offset = trainer.load_counters()[1]

    #################### Training Phase ######################
    for i, (train_dataset, val_dataset) in track(enumerate(k_fold.split(dataset)), "[cyan]Training...", total=5):
        #if i < offset: continue
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        trainer.fit(train_loader, val_loader, num_epochs=epochs)
        trainer.test(val_loader)

    print('[bold green]Training completed!')
