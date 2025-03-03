import pandas as pd

column_names = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
df = pd.read_csv("raw/dontpatronizeme.tsv", sep='\t', header=None, names=column_names)

train_parids_df = pd.read_csv("raw/train-parids.csv")
dev_parids_df = pd.read_csv("raw/dev-parids.csv")

# Number of rows in the entire dataset should be split across train and dev
assert train_parids_df.shape[0] + dev_parids_df.shape[0] == df.shape[0]

# Par IDs should be unique within train and dev
assert train_parids_df["par_id"].is_unique
assert dev_parids_df["par_id"].is_unique

# Par IDs should not overlap
assert not train_parids_df["par_id"].isin(dev_parids_df["par_id"]).any()

train_df= df[df["par_id"].isin(train_parids_df["par_id"])]
dev_df= df[df["par_id"].isin(dev_parids_df["par_id"])]

train_df.to_csv("train.csv", index=False)
dev_df.to_csv("dev.csv", index=False)
