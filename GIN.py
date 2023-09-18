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
from torch.optim import Adam, Adamax
import numpy as np
from rich import print
from rich.progress import track
import argparse as ap
import os

parser = ap.ArgumentParser()
parser.add_argument("-e", "--epochs", default=100, help="Epochs number")
parser.add_argument("-d", "--dir", default="training", help="Insert the train path")
parser.add_argument("-t", "--test", action="store_true", help="Generate test file for the Confusion Matrix")
args = vars(parser.parse_args())

############## Properites #################
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256
features = ["de_movingAve"]
lr = 0.001
weight_decay = 0.0
epochs = int(args["epochs"])
train_directory = str(args["dir"])

#################### Classification Trainer ######################
class MyClassificationTrainer(ClassificationTrainer):
    def __init__(self, model: nn.Module, trainer_k:str=None, lr:float=0.001, weight_decay:float=0, num_classes=None, optimizer=None, device_ids: List[int] = [], ddp_sync_bn: bool = True, ddp_replace_sampler: bool = True, ddp_val: bool = True, ddp_test: bool = True):
        super().__init__(model, num_classes, lr, weight_decay, device_ids, ddp_sync_bn, ddp_replace_sampler, ddp_val, ddp_test)
        self.writer = SummaryWriter("{0}/train{1}/loss".format(train_directory, trainer_k))
        self.steps_file_name = "{0}/train{1}/steps".format(train_directory, trainer_k)
        try:
            model.load_state_dict(torch.load("{0}/train{1}/model.pth".format(train_directory, trainer_k)))
        except FileNotFoundError:
            pass
        self.train_counter = 0
        self.epoch = 0
        self.optimizer = optimizer
        self.test_counter = 0
        self.trainer_k = trainer_k
        self.last_train_loss = 0
        self.last_train_accuracy = 0
        self.cmdata_file = open("{0}/train{1}/metrics_data".format(train_directory, trainer_k), "a+")

    def on_training_step(self, train_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        super().on_training_step(train_batch, batch_id, num_batches, **kwargs)
        if self.train_loss.mean_value.item() != 0:
            self.last_train_loss = self.train_loss.compute()
            self.last_train_accuracy = self.train_accuracy.compute()

    def after_validation_epoch(self, epoch_id: int, num_epochs: int,**kwargs):
        super().after_validation_epoch(epoch_id, num_epochs, **kwargs)
        torch.save(model.state_dict(), "{0}/train{1}/model.pth".format(train_directory, self.trainer_k))
        self.writer.add_scalars('loss', {
            'train': self.last_train_loss,
            'validation': self.val_loss.compute()
        }, self.train_counter)
        self.writer.add_scalars('accuracy', {
            'train': self.last_train_accuracy*100,
            'validation': self.val_accuracy.compute()*100
        }, self.train_counter)
        self.train_counter += 1
        self.epoch += 1

    def after_test_epoch(self, **kwargs):
        super().after_test_epoch(**kwargs)
        self.writer.add_scalar("loss/test", self.test_loss.compute(), self.epoch)
        self.writer.add_scalar("accuracy/test", self.test_accuracy.compute()*100, self.epoch)
        self.test_counter += 1

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int, **kwargs):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)
        pred = self.modules['model'](X)
        
        data = np.column_stack((pred.tolist(), y))
        for row in data:
            line = ", ".join([f"{elem}" for elem in row])
            self.cmdata_file.write(f"{line}\n")

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_accuracy.update(pred.argmax(1), y)

if __name__ == '__main__':
    #################### 16 CHANNELS EXTRACTION ######################
    """UMIV_CHANNEL_LIST = ['FP1', 'FP2', 'F7', 'F3', 'F4',
                         'F8', 'T7', 'C3', 'C4', 'T8', 'P7',
                         'P3', 'P4', 'P8', 'O1', 'O2']
    
    SEED_IV_ADJACENCY_MATRIX = format_adj_matrix_from_adj_list(UMIV_CHANNEL_LIST, SEED_IV_ADJACENCY_LIST)
    """

    ################## Load Features Dataset ######################
    dataset = SEEDIVFeatureDataset(io_path=f'./dataset/seed_iv',
                      root_path='./dataset/eeg_feature_smooth',
                      feature=features,
                      online_transform=transforms.Compose([transforms.MeanStdNormalize(),ToG(SEED_IV_ADJACENCY_MATRIX)]),
                      label_transform=transforms.Select('emotion'))


    #################### Init Trainer ######################
    k_fold = KFoldGroupbyTrial(n_splits=5, shuffle=True, random_state=10, split_path='./dataset/split/')

    #################### Training Phase ######################
    for i, (train_dataset, val_dataset) in track(enumerate(k_fold.split(dataset)), "[cyan]Training...", total=5):
        # Initialize the GIN model
        model = GIN(in_channels=5, hid_channels=64, num_classes=4).to(device=device)
        # Initialize trainer, you can change the optimizer with Adam
        trainer = MyClassificationTrainer(model=model, trainer_k=i, optimizer=Adam(model.parameters(),lr=lr, weight_decay=weight_decay))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        if not args["test"]:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
                trainer.fit(train_loader, val_loader, num_epochs=epochs)
        trainer.test(val_loader)

    print('[bold green]Training completed!')
