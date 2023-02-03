from pprint import pprint
from label_studio_ml.model import LabelStudioMLBase

import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from label_studio_ml.model import LabelStudioMLBase
import pandas as pd
from label_studio_sdk.project import Project
from label_studio_sdk import Client
from sklearn.model_selection import train_test_split
from baal.modelwrapper import ModelWrapper
from torchvision.models import efficientnet_b0
from baal.bayesian.dropout import MCDropoutModule
from baal.active import FileDataset, ActiveLearningDataset
import torchvision
from baal.active.heuristics import BALD
from torchvision.models import vgg16
import warnings
from collections import Counter
from dataclasses import dataclass
from pickle import TRUE
from pprint import pprint
from typing import Any, Callable, List, Union
from matplotlib import pyplot
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import albumentations as A

from torch.utils.data import DataLoader2
from torchvision.transforms import transforms
from loguru import logger
from torchvision.datasets import ImageFolder
from torch.cuda.amp.autocast_mode import autocast

from tqdm import tqdm
import numpy as np
import copy
from pytorch_lightning.loggers import MLFlowLogger

from pytorch_lightning.callbacks import ModelCheckpoint


import torchvision

from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import MulticlassPrecisionRecallCurve

from torchmetrics.classification.confusion_matrix import (
    ConfusionMatrix,
    MulticlassConfusionMatrix,
)
import pandas as pd
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.metrics import accuracy_score

import awswrangler as wr
from sklearn.model_selection import train_test_split
from aum import AUMCalculator
import os
import cleanlab_filter
import itertools
from cleanlab.outlier import OutOfDistribution
import glob
import awswrangler as wr
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import splitfolders
from torchvision.datasets import ImageFolder
import cv2
from uuid import uuid4
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


class ImageDataset2(Dataset):
    def __init__(self, df: pd.DataFrame, partName: str, transform: Callable):
        self.df = df
        self.transform = transform
        self.partName = partName
        # self.labelIdx = {"not_dmg": 0, "dmg": 1, "slight_dmg": 1, "heavy_dmg": 1}
        self.imgDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]

        labelTensor = torch.tensor(row[self.partName], dtype=torch.long)
        image = cv2.imread(os.path.join(self.imgDir, filename))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)

        data = {
            "img": sample["image"],
            "target": labelTensor,
            "file": filename,
        }
        return data


def get_dataloader(trainDf: pd.DataFrame, valDf: pd.DataFrame, partName: str):

    batchSize = 8
    trainCPUWorker = 0
    imgSize = 480

    trainTransform = A.Compose(
        [
            A.LongestMaxSize(imgSize),
            A.PadIfNeeded(
                min_height=imgSize,
                min_width=imgSize,
                border_mode=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5, brightness=0.8, saturation=0.8, hue=0.5, contrast=0.8),
            A.Rotate(limit=180, border_mode=0, p=0.3),
            A.RandomGridShuffle(grid=(2, 2), p=0.3),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            # A.Rotate(border_mode=0, p=0.2),
            # A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # A.Downscale(scale_min=0.6, scale_max=0.8, p=0.2),
            # A.Normalize(),
            ToTensorV2(),
        ]
    )
    evalTransform = A.Compose(
        [
            A.LongestMaxSize(imgSize),
            A.PadIfNeeded(
                min_height=imgSize,
                min_width=imgSize,
                border_mode=0,
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            # A.Normalize(),
            ToTensorV2(),
        ]
    )
    trainDs = ImageDataset2(df=trainDf, transform=trainTransform, partName=partName)

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=batchSize,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )
    evalDs = ImageDataset2(df=valDf, transform=evalTransform, partName=partName)
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=batchSize,
        num_workers=trainCPUWorker,
        persistent_workers=False,
    )
    testDs = ImageDataset2(df=valDf, transform=evalTransform, partName=partName)
    testLoader = DataLoader2(
        testDs,
        shuffle=False,
        batch_size=batchSize,
        num_workers=trainCPUWorker,
        persistent_workers=False,
    )
    allTrainSamples = [x for x in trainLoader.dataset.df["filename"].tolist()]
    allValSamples = [x for x in evalLoader.dataset.df["filename"].tolist()]
    allTestSamples = [x for x in evalLoader.dataset.df["filename"].tolist()]

    assert set(allTrainSamples).isdisjoint(set(allValSamples))
    assert set(allTrainSamples).isdisjoint(set(allTestSamples))

    return trainLoader, evalLoader, testLoader


def create_model():

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT,
        # weight=None
    )
    # for p in model.parameters():
    #     p.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=2)
    # for p in model.fc.parameters():
    #     p.requires_grad = True

    return model


def train_eval(trainLoader, valLoader, testLoader, partName, version):

    checkpoint_callback = ModelCheckpoint(
        monitor="e_acc",
        save_top_k=1,
        mode="max",
        filename="{e_acc:.2f}-{tp_diff:.2f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="e_acc",
        stopping_threshold=0.95,
        patience=10,
        verbose=True,
        min_delta=0.01,
        mode="max",
    )
    trainProcessModel = ProcessModel(partName)
    logDir = f"/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/speedup/{partName}"
    os.makedirs(logDir, exist_ok=True)
    trainer1 = pl.Trainer(
        # accumulate_grad_batches=10,
        default_root_dir=logDir,
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=20,
        num_sanity_val_steps=0,
        benchmark=True,
        precision=16,
        # logger=logger,
        log_every_n_steps=30,
        # auto_scale_batch_size="binsearch",
        callbacks=[early_stop_callback],
        detect_anomaly=False,
        enable_model_summary=False,
        enable_checkpointing=False
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    trainer1.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    return trainProcessModel.testAccVal


class ProcessModel(pl.LightningModule):
    def __init__(self, part):
        super(
            ProcessModel,
            self,
        ).__init__()
        pl.seed_everything(99)

        self.learning_rate = 1e-3
        self.part = part

        # self.save_hyperparameters(ignore="model")
        self.model = create_model()

        self.testAccMetric = MulticlassAccuracy(num_classes=2)
        self.trainAccMetric = MulticlassAccuracy(num_classes=2)

        self.testConfMat = MulticlassConfusionMatrix(
            num_classes=2, normalize="true"
        ).to(self.device)
        self.testF1 = F1Score(task="multiclass", num_classes=2).to(self.device)
        self.trainConfMat = MulticlassConfusionMatrix(
            num_classes=2, normalize="true"
        ).to(self.device)

        self.testPrecision = Precision(
            task="multiclass",
            num_classes=2,
        ).to(self.device)
        self.testRecall = Recall(task="multiclass", num_classes=2).to(self.device)
        # if isFiltered:
        #     self.clsWeight = torch.tensor([1.0, 2.0])
        # else:
        # self.clsWeight = torch.tensor([1.0, 1.0])

        self.criterion = torch.nn.CrossEntropyLoss()
        self.predCriterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.testAccVal = 0.5
        # self.pr_curve = MulticlassPrecisionRecallCurve(num_classes=2, thresholds=11)

    def configure_optimizers(self):

        return torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
        )

    def forward(self, imgs):
        logit = self.model(imgs)
        return logit

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        targets = batch["target"]
        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, dim=1)

        # self.trainAccMetric.update(preds, targets)
        # self.trainConfMat.update(preds, targets)
        self.log("train_loss", loss)
        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     testAcc = self.trainAccMetric.compute()
    #     self.log("t_acc", testAcc, prog_bar=True)
    #     self.trainAccMetric.reset()
    #     confMat = self.trainConfMat.compute()

    #     self.log("tp_0", confMat[0][0], prog_bar=False)
    #     self.log("tp_1", confMat[1][1], prog_bar=False)
    #     # self.log("tp_2", confMat[2][2], prog_bar=False)

    #     self.trainConfMat.reset()

    #     return super().training_epoch_end(outputs)

    def validation_step(self, batch, idx):
        imgs = batch["img"]
        targets = batch["target"]
        images = imgs.to(self.device)
        targets = targets.to(self.device)
        logit = self.model(images)
        loss = self.criterion(logit, targets)
        preds = torch.argmax(logit, dim=1)
        self.testAccMetric.update(preds, targets)
        self.testConfMat.update(preds, targets)
        self.testPrecision.update(preds, targets)
        self.testRecall.update(preds, targets)
        self.testF1.update(preds, targets)

        self.log("e_loss", loss, prog_bar=False)

        return preds, targets

    def validation_epoch_end(self, val_step_outputs) -> None:
        testAcc = self.testAccMetric.compute()
        testPrecision = self.testPrecision.compute()
        testRecall = self.testRecall.compute()
        testF1 = self.testF1.compute()
        self.log("e_acc", testAcc, prog_bar=True)
        self.log("precision", testPrecision, prog_bar=False)
        self.log("recall", testRecall, prog_bar=False)
        self.log("f1", testF1, prog_bar=True)

        confMat = self.testConfMat.compute()

        self.log("tp_0", confMat[0][0], prog_bar=True)
        self.log("tp_1", confMat[1][1], prog_bar=True)
        tpDiff = torch.abs(confMat[1][1] - confMat[0][0])
        self.log("tp_diff", tpDiff, prog_bar=True)

        self.testAccVal = testAcc

        self.testConfMat.reset()
        self.testRecall.reset()
        self.testPrecision.reset()
        self.testAccMetric.reset()
        self.testF1.reset()

        return super().validation_epoch_end(val_step_outputs)


def tmc(partName: str, allAvailableDf: pd.DataFrame, testDf: pd.DataFrame):
    iterations = 200
    stopThreshold = 0.7
    contribList = []
    startingBatch = 40
    truncationCount = 0
    testDf = testDf.groupby(partName).sample(n=100)
    outputDir = "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/output"
    runId = uuid4().hex
    for it in tqdm(range(iterations), desc="iterations"):
        allAvailableDf = allAvailableDf.sample(frac=1)
        shapTrainDf = allAvailableDf.groupby(partName).sample(n=20)
        allAvailableDf = allAvailableDf[
            ~allAvailableDf["filename"].isin(shapTrainDf["filename"].tolist())
        ]
        oldScore = 0.5
        for i in tqdm(range(startingBatch, 200, 10), desc="batch"):
            newSubset = allAvailableDf.iloc[i : i + 10]
            shapTrainDf = pd.concat([shapTrainDf, newSubset])
            shapTrainDf.drop_duplicates(subset="filename", inplace=True)

            trainLoader, valLoader, testLoader = get_dataloader(
                shapTrainDf, testDf, partName
            )
            newScore = train_eval(trainLoader, valLoader, testLoader, partName, "1")
            contrib = newScore - oldScore
            if not isinstance(contrib, float):
                contrib = contrib.cpu().numpy()
            if i == startingBatch:
                contrib = contrib / 10
            logger.success(f"Curr Acc : {newScore} Contrib : {contrib}")
            # contrib = contrib.cpu().numpy()
            # sampleId = allAvailableDf.iloc[[i]]["filename"].values[0]

            contribList.extend(
                [
                    {
                        "filename": x["filename"],
                        "contrib": contrib,
                        "curr_acc": float(newScore.cpu().numpy()),
                    }
                    for _, x in newSubset.iterrows()
                ]
            )
            oldScore = newScore
            if oldScore >= stopThreshold:
                truncationCount += 1
                if truncationCount >= 5:
                    break
            if i % 100 == 0:
                contribDf = pd.json_normalize(contribList)
                partDir = f"{outputDir}/{partName}"
                os.makedirs(partDir, exist_ok=True)
                contribDf.to_csv(f"{partDir}/{partName}_sample_contrib_{runId}.csv")
                print(pd.json_normalize(contribList))


if __name__ == "__main__":
    partName = "bonnet"

    testDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/train_val/bonnet/clean_test.csv"
    )
    allAvailableDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/train_val/bonnet/train_shapley.csv"
    )
    labelIssueDf = pd.read_parquet(
        "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/train_val/bonnet/label_issue.parquet"
    )
    allAvailableDf = allAvailableDf[
        allAvailableDf["filename"].isin(labelIssueDf["filename"].tolist())
    ]
    allAvailableDf = allAvailableDf[~allAvailableDf["filename"].isin(testDf)]

    allAvailableDf = allAvailableDf.groupby(partName).sample(n=1000)
    Parallel(n_jobs=10, prefer="threads")(
        delayed(tmc)(partName, allAvailableDf, testDf) for i in range(100)
    )
