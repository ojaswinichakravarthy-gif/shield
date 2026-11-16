import pickle

from config import FEATURES_PATH, PROCESSED_DATA_PATH

with open(PROCESSED_DATA_PATH, "rb") as f:
    processed = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    eng = pickle.load(f)

df = eng["df"]

train_df = df.iloc[processed["train_idx"]]
test_df = df.iloc[processed["test_idx"]]

train_accounts = (
    set(train_df["sender"])
    | set(train_df["receiver"])
)

test_accounts = (
    set(test_df["sender"])
    | set(test_df["receiver"])
)

overlap = train_accounts & test_accounts

print("Train Accounts:", len(train_accounts))
print("Test Accounts:", len(test_accounts))
print("Overlap:", len(overlap))
print("Overlap %:", len(overlap)/len(test_accounts)*100)

print(df.groupby("isFraud")["sender"].nunique())
print(df.groupby("isFraud")["receiver"].nunique())