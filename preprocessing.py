import pandas as pd

column_names = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
df = pd.read_csv("data/raw/dontpatronizeme.tsv", sep='\t', header=None, names=column_names)

# Convert to binary classification problem
df["label"] = (df["label"] >= 2).astype(int)

train_parids_df = pd.read_csv("data/raw/train-parids.csv")
dev_parids_df = pd.read_csv("data/raw/dev-parids.csv")

# Number of rows in the entire dataset should be split across train and dev
assert train_parids_df.shape[0] + dev_parids_df.shape[0] == df.shape[0]

# Par IDs should be unique within train and dev
assert train_parids_df["par_id"].is_unique
assert dev_parids_df["par_id"].is_unique

# Par IDs should not overlap
assert not train_parids_df["par_id"].isin(dev_parids_df["par_id"]).any()

dev_parids_df.drop(columns=["label"])

train_df= df[df["par_id"].isin(train_parids_df["par_id"])]
dev_df = dev_parids_df.merge(df, on="par_id", how="left")
dev_df = dev_df.rename(columns={"label_y": "label"})

train_df.to_csv("data/train.csv", index=False)
dev_df.to_csv("data/dev.csv", index=False)
df.to_csv("data/complete.csv", index=False)
