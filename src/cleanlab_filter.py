from cleanlab.filter import find_label_issues
import pandas as pd
import numpy as np

if __name__ == "__main__":
    predDf = pd.read_csv(
        "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/output/confidence_filtering/pred.csv"
    )
    outputDir = "/home/alextay96/Desktop/all_workspace/new_workspace/ML_Data_Debug_Tools/data/train_val/bonnet"
    print(predDf["softmax_probs"])
    softmaxProbs = np.array([eval(x) for x in predDf["softmax_probs"].tolist()])
    ordered_label_issues = find_label_issues(
        labels=predDf["gt"],
        pred_probs=softmaxProbs,
        return_indices_ranked_by="self_confidence",
        filter_by="both",
        frac_noise=0.5,
    )
    issueDf = predDf.iloc[ordered_label_issues]
    issueDf["filename"] = issueDf["file"].apply(lambda x: x.split("/")[-1])
    print(issueDf)
    issueDf.to_parquet(f"{outputDir}/label_issue_2.parquet")
