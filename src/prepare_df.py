import pandas as pd
import glob
import os
import awswrangler as wr

wr.config.s3_endpoint_url = "http://localhost:8333"

multilabelDf = wr.s3.read_parquet(
    path=f"s3://multilabel_df/",
    dataset=True,
)
print(multilabelDf.columns)
srcDir = "/home/alextay96/Desktop/all_workspace/new_workspace/DLDataPipeline/data/build_dataset/pred_multilabel_aum_v2"
allFilesDf = pd.concat(
    [pd.read_csv(x) for x in glob.glob(f"{srcDir}/**/*.csv", recursive=True)]
)
print(allFilesDf.columns)
allParts = os.listdir(
    "/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/data/labels_final"
)
allParts = sorted(allParts)
print(allParts)
partName = allParts[0]
print(partName)
testSetDf = pd.read_csv(
    f"/home/alextay96/Desktop/all_workspace/new_workspace/mrm_active_learning/data/labels_final/{partName}/complete_label.csv"
)
# print(testSetDf["example"])
testSetDf["filename"] = testSetDf["example"].apply(lambda x: x.split("/")[-1])
testSetDf["CaseID"] = testSetDf["filename"].apply(lambda x: int(x.split("_")[0]))

print(testSetDf.columns)
targetViews = ["Front View", "Front View Left", "Front View Right"]
trainXDf = allFilesDf[
    (allFilesDf["view"].isin(targetViews))
    & (~allFilesDf["filename"].isin(testSetDf["filename"].unique().tolist()))
]
trainXDf["CaseID"] = trainXDf["filename"].apply(lambda x: int(x.split("_")[0]))
trainDf = trainXDf.merge(multilabelDf[["CaseID", partName]], on="CaseID")
trainDf = trainDf[["CaseID", "filename", partName]]
testSetDf[partName] = testSetDf["label"].apply(lambda x: 1 if x == "dmg" else 0)
testSetDf.drop_duplicates(subset="filename", inplace=True)
testSetDf = testSetDf[["CaseID", "filename", partName]]
outputDir = f"/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/train_val/{partName}"
os.makedirs(outputDir, exist_ok=True)
testSetDf.to_csv(f"{outputDir}/clean_test.csv")
trainDf.to_csv(f"{outputDir}/train_shapley.csv")

print(testSetDf)
