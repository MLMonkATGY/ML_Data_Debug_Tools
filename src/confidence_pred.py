import pandas as pd
from train_1 import ImageDataset, ProcessModel
import pandas as pd
import os
from pprint import pprint
from train_1 import ProcessModel
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
from torch.cuda.amp.autocast_mode import autocast
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

if __name__ == "__main__":
    mapping = {
        "bonnet": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/bonnet/lightning_logs/version_4/checkpoints/e_acc=0.83-tp_diff=0.00.ckpt",
        "bumper front": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/bumper_front/lightning_logs/version_2/checkpoints/e_acc=0.88-tp_diff=0.07.ckpt",
        "bumper rear": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/bumper_rear/lightning_logs/version_1/checkpoints/e_acc=0.89-tp_diff=0.03.ckpt",
        "engine": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/engine/lightning_logs/version_1/checkpoints/e_acc=0.89-tp_diff=0.03.ckpt",
        "fender front lh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/fender_front/lightning_logs/version_1/checkpoints/e_acc=0.81-tp_diff=0.05.ckpt",
        "fender front rh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/fender_front/lightning_logs/version_1/checkpoints/e_acc=0.81-tp_diff=0.05.ckpt",
        "front panel": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/front_panel/lightning_logs/version_1/checkpoints/e_acc=0.87-tp_diff=0.00.ckpt",
        "grille": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/grille/lightning_logs/version_2/checkpoints/e_acc=0.93-tp_diff=0.08.ckpt",
        "headlamp lh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/headlamp/lightning_logs/version_2/checkpoints/e_acc=0.86-tp_diff=0.04.ckpt",
        "headlamp rh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/headlamp/lightning_logs/version_2/checkpoints/e_acc=0.86-tp_diff=0.04.ckpt",
        "rear compartment": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/rear_compartment/lightning_logs/version_1/checkpoints/e_acc=0.84-tp_diff=0.07.ckpt",
        "rear panel": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/rear_panel/lightning_logs/version_1/checkpoints/e_acc=0.88-tp_diff=0.08.ckpt",
        "rear quarter lh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/rear_quarter/lightning_logs/version_1/checkpoints/e_acc=0.83-tp_diff=0.01.ckpt",
        "rear quarter rh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/rear_quarter/lightning_logs/version_1/checkpoints/e_acc=0.83-tp_diff=0.01.ckpt",
        "tail lamp lh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/tail_lamp/lightning_logs/version_2/checkpoints/e_acc=0.88-tp_diff=0.02.ckpt",
        "tail lamp rh": "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/output/clean/tail_lamp/lightning_logs/version_2/checkpoints/e_acc=0.88-tp_diff=0.02.ckpt",
    }
    unlabelDf = {
        "bonnet": "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/train_val/bonnet/train_shapley.csv"
    }
    outputDir = "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/output/confidence_filtering"
    os.makedirs(outputDir, exist_ok=True)
    allMetrics = []
    imgSize = 640
    batchSize = 40
    device = torch.device("cuda")
    allPredDf = []
    topNCaseNum = 5000
    for part, modelPath in mapping.items():
        model = ProcessModel.load_from_checkpoint(modelPath)
        model.eval()
        model = model.to(device)
        partCsv = unlabelDf[part]
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
        testDf = pd.read_csv(partCsv)
        # testDf = testDf[testDf["CaseID"].isin(unseenCaseID)]
        # print(testDf)
        print(len(testDf["CaseID"].unique()))
        testDs = ImageDataset(
            df=testDf,
            # partname=part,
            transform=evalTransform,
        )
        testLoader = DataLoader2(
            testDs,
            shuffle=False,
            batch_size=batchSize * 3,
            num_workers=10,
            persistent_workers=False,
        )
        testConfMat = MulticlassConfusionMatrix(num_classes=2, normalize="true").to(
            device
        )
        testF1 = F1Score(task="multiclass", num_classes=2).to(device)
        testPrecision = Precision(
            task="multiclass",
            num_classes=2,
        ).to(device)
        testRecall = Recall(task="multiclass", num_classes=2).to(device)
        testAccMetric = MulticlassAccuracy(num_classes=2).to(device)
        partMetrics = {}
        batchIdx = 0
        with torch.no_grad():
            for batch in tqdm(testLoader):
                batchIdx += 1
                imgs = batch["img"]
                targets = batch["target"]
                file = batch["file"]
                # caseID = batch["CaseID"]
                # view = batch["view"]

                images = imgs.to(device)
                targets = targets.to(device)
                with autocast():
                    logit = model(images)
                    # preds = torch.argmax(logit, dim=1)
                    predProbs, preds = torch.max(torch.softmax(logit, dim=1), dim=1)
                    softMaxProbs = torch.softmax(logit, dim=1).cpu().numpy().tolist()
                predDf = pd.DataFrame(
                    {
                        "file": file,
                        "pred": preds.cpu().numpy(),
                        "gt": targets.cpu().numpy(),
                        # "CaseID": caseID.cpu().numpy(),
                        # "view": view,
                        "pred_probs": predProbs.cpu().numpy(),
                        # "softmax_probs": softMaxProbs.cpu().numpy(),
                    }
                )
                # print(predDf.index)
                predDf["tempId"] = list(range(len(predDf)))
                predDf["softmax_probs"] = predDf.apply(
                    lambda x: softMaxProbs[x["tempId"]], axis=1
                )
                predDf.drop(columns="tempId", inplace=True)
                predDf["part"] = part
                if batchIdx % 50 == 0:
                    tempPred = pd.concat(allPredDf)
                    tempPred.to_csv(f"{outputDir}/pred.csv")
                allPredDf.append(predDf)
                testAccMetric.update(preds, targets)
                testConfMat.update(preds, targets)
                testPrecision.update(preds, targets)
                testRecall.update(preds, targets)
                testF1.update(preds, targets)

        testAcc = testAccMetric.compute()
        testPrecision = testPrecision.compute()
        testRecall = testRecall.compute()
        testF1 = testF1.compute()
        confMat = testConfMat.compute()

        partMetrics["f1"] = testF1
        partMetrics["acc"] = testAcc
        partMetrics["tp_0"] = confMat[0][0]
        partMetrics["tp_1"] = confMat[1][1]
        partMetrics["recall"] = testRecall
        partMetrics["precision"] = testPrecision
        partMetrics["part"] = part
        pprint(partMetrics)
        allMetrics.append(partMetrics)
    completePredDf = pd.concat(allPredDf)
    completePredDf.to_csv(f"{outputDir}/pred.csv")
    metrics = pd.json_normalize(allMetrics)
    metrics.to_csv(f"{outputDir}/part_metrics.csv")
