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
from torch.cuda.amp import autocast
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
import itertools
from cleanlab.outlier import OutOfDistribution
import glob
import awswrangler as wr
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import splitfolders
from torchvision.datasets import ImageFolder
import cv2

warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        self.df = df
        self.transform = transform
        self.imgDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgPath = os.path.join(self.imgDir, row["filename"])
        target = row["bonnet"]
        labelTensor = torch.tensor(target, dtype=torch.long)
        image = cv2.imread(imgPath)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)

        data = {
            "img": sample["image"],
            "target": labelTensor,
            "file": imgPath,
        }
        return data


class ImageDataset2(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        self.df = df
        self.transform = transform
        self.labelIdx = {"not_dmg": 0, "dmg": 1, "slight_dmg": 1, "heavy_dmg": 1}
        # self.imgDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/imgs"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgPath = row["example"]
        target = self.labelIdx[row["label"]]
        labelTensor = torch.tensor(target, dtype=torch.long)
        image = cv2.imread(imgPath)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample = self.transform(image=image)

        data = {
            "img": sample["image"],
            "target": labelTensor,
            "file": imgPath,
        }
        return data


def get_dataloader(trainDf: pd.DataFrame, valDf: pd.DataFrame):

    batchSize = 20
    trainCPUWorker = 6
    imgSize = 640

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
            A.Rotate(limit=180, border_mode=0, p=0.2),
            A.RandomGridShuffle(grid=(2, 2), p=0.2),
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
    trainDs = ImageDataset2(
        df=trainDf,
        transform=trainTransform,
    )

    trainLoader = DataLoader2(
        trainDs,
        shuffle=True,
        batch_size=batchSize,
        num_workers=trainCPUWorker,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    evalDs = ImageDataset2(
        df=valDf,
        transform=evalTransform,
    )
    evalLoader = DataLoader2(
        evalDs,
        shuffle=False,
        batch_size=batchSize * 2,
        num_workers=trainCPUWorker,
        persistent_workers=True,
    )
    testDs = ImageDataset2(
        df=valDf,
        transform=evalTransform,
    )
    testLoader = DataLoader2(
        testDs,
        shuffle=False,
        batch_size=batchSize * 2,
        num_workers=trainCPUWorker,
        persistent_workers=True,
    )
    allTrainSamples = [x for x in trainLoader.dataset.df["example"].tolist()]
    allValSamples = [x for x in evalLoader.dataset.df["example"].tolist()]
    allTestSamples = [x for x in evalLoader.dataset.df["example"].tolist()]

    assert set(allTrainSamples).isdisjoint(set(allValSamples))
    assert set(allTrainSamples).isdisjoint(set(allTestSamples))

    return trainLoader, evalLoader, testLoader


def create_model():

    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT,
        # weight=None
    )

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=2)

    return model


def create_new_label_task(
    predDf: pd.DataFrame, allSampleDf: pd.DataFrame, outputDir: str, version: str
):
    predDf.sort_values(by="probs", ascending=True, inplace=True)
    topSamplesDf = predDf.head(300)
    topSamplesDf["file"] = topSamplesDf["files"].apply(lambda x: x.split("/")[-1])
    samplesToLabel = allSampleDf[
        allSampleDf["file"].isin(topSamplesDf["file"].tolist())
    ]
    urls = "\n".join(samplesToLabel["filepath"].tolist())
    ds_v2 = f"{outputDir}/{version}_dataset_by_probs.txt"
    with open(ds_v2, "w") as f:
        f.write(urls)
    print(urls)


def train_eval(trainLoader, valLoader, testLoader, partName, version):

    checkpoint_callback = ModelCheckpoint(
        monitor="e_acc",
        save_top_k=3,
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
    logDir = f"/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/{partName}"
    os.makedirs(logDir, exist_ok=True)
    trainer1 = pl.Trainer(
        # accumulate_grad_batches=10,
        default_root_dir=logDir,
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        benchmark=True,
        precision=16,
        # logger=logger,
        log_every_n_steps=30,
        # auto_scale_batch_size="binsearch",
        callbacks=[checkpoint_callback, early_stop_callback],
        detect_anomaly=False,
        # limit_train_batches=1,
        # limit_val_batches=1,
    )

    trainer1.fit(
        trainProcessModel, train_dataloaders=trainLoader, val_dataloaders=valLoader
    )
    # allPredDf = trainer1.predict(trainProcessModel, dataloaders=testLoader)
    # completePredDf: pd.DataFrame = pd.concat(allPredDf)
    # completePredDf.sort_values(by="probs", ascending=True, inplace=True)
    # completePredDf.to_csv(f"{logDir}/hard_dmg_{version}.csv")
    # return completePredDf, logDir
    # analyse_loss(completePredDf)


class ProcessModel(pl.LightningModule):
    def __init__(self, part):
        super(
            ProcessModel,
            self,
        ).__init__()
        pl.seed_everything(99)

        self.learning_rate = 1e-3
        self.part = part

        self.save_hyperparameters(ignore="model")
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
        # self.pr_curve = MulticlassPrecisionRecallCurve(num_classes=2, thresholds=11)

    def configure_optimizers(self):

        return torch.optim.Adam(
            self.parameters(),
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

        self.trainAccMetric.update(preds, targets)
        self.trainConfMat.update(preds, targets)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        testAcc = self.trainAccMetric.compute()
        self.log("t_acc", testAcc, prog_bar=True)
        self.trainAccMetric.reset()
        confMat = self.trainConfMat.compute()

        self.log("tp_0", confMat[0][0], prog_bar=False)
        self.log("tp_1", confMat[1][1], prog_bar=False)
        # self.log("tp_2", confMat[2][2], prog_bar=False)

        self.trainConfMat.reset()

        return super().training_epoch_end(outputs)

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

        self.testConfMat.reset()
        self.testRecall.reset()
        self.testPrecision.reset()
        self.testAccMetric.reset()
        self.testF1.reset()

        return super().validation_epoch_end(val_step_outputs)

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
    #     imgs = batch["img"]
    #     targets = batch["target"]
    #     imgFiles = batch["file"]
    #     images = imgs.to(self.device)
    #     targets = targets.to(self.device)
    #     posTarget = torch.tensor([1] * len(targets)).to(self.device)
    #     negTarget = torch.tensor([0] * len(targets)).to(self.device)

    #     logit = self.model(images)
    #     lossPerSample = self.predCriterion(logit, posTarget)
    #     pred_probs, preds = torch.max(torch.softmax(logit, dim=1), dim=1)

    #     invTarget = 1 - targets
    #     invLossPerSample = self.predCriterion(logit, negTarget)
    #     predDf = pd.DataFrame(
    #         {
    #             "files": imgFiles,
    #             "pos_loss": lossPerSample.cpu().numpy(),
    #             "neg_Loss": invLossPerSample.cpu().numpy(),
    #             "gt": targets.cpu().numpy(),
    #             "pred": preds.cpu().numpy(),
    #             "probs": pred_probs.cpu().numpy(),
    #         }
    #     )
    #     return predDf


def get_unlabelled_dataset(labelledDf: pd.DataFrame, serveFilesPath: str):
    with open(serveFilesPath, "r") as f:
        allServeFileUrl = f.read().split("\n")
    allSampleDf = pd.DataFrame({"filepath": allServeFileUrl})
    allSampleDf["cat"] = allSampleDf["filepath"].apply(lambda x: x.split("/")[-2])
    allSampleDf["file"] = allSampleDf["filepath"].apply(lambda x: x.split("/")[-1])
    hardDmgSample = allSampleDf[allSampleDf["cat"] == "hard_dmg"]

    allLabelledFiles = labelledDf["file"].tolist()
    unlabelledFilename = set(hardDmgSample["file"].tolist()).difference(
        set(allLabelledFiles)
    )
    ds_v2 = pd.DataFrame({"file": list(unlabelledFilename)})
    ds_v2["label"] = 1
    return ds_v2, allSampleDf


def get_label_df():
    labelCsv = [
        "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/data/label_studio/labels/rear_quarter_clean_v1.csv",
        "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/data/label_studio/labels/rear_quarter_clean_v2.csv",
    ]
    rawDf = pd.concat([pd.read_csv(x) for x in labelCsv])
    rawDf.drop_duplicates(subset="file", inplace=True)
    return rawDf


if __name__ == "__main__":
    allParts = os.listdir(
        "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/data/labels_final"
    )
    # allParts = ["bonnet"]
    for partName in allParts:
        version = "v1"
        labelDf = pd.read_csv(
            f"/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/data/labels_final/{partName}/complete_label.csv"
        )
        labelDf.drop_duplicates(subset="example", inplace=True)
        trainDf, valDf = train_test_split(labelDf)

        trainLoader, valLoader, testLoader = get_dataloader(trainDf, valDf)
        train_eval(trainLoader, valLoader, testLoader, partName, version)
