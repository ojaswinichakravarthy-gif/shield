import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("data/dataset 2/creditcard.csv")

print("=" * 60)
print("Credit Card Fraud Dataset")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True))

# ==========================================================
# 2. SPLIT FEATURES AND TARGET
# ==========================================================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================================
# 3. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("After Train/Test Split")
print("=" * 60)

print("\nTraining Set Shape:")
print(X_train.shape)

print("\nTest Set Shape:")
print(X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTest Class Distribution:")
print(y_test.value_counts())

# ==========================================================
# 4. APPLY SMOTE ONLY TO TRAINING DATA
# ==========================================================

print("\n" + "=" * 60)
print("Applying SMOTE To Training Set")
print("=" * 60)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining Distribution AFTER SMOTE:")
print(y_train_smote.value_counts())

print("\nTraining Shape AFTER SMOTE:")
print(X_train_smote.shape)

print("\nTest Distribution (UNCHANGED):")
print(y_test.value_counts())

# ==========================================================
# 5. SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"Original Training Samples : {len(X_train):,}")
print(f"SMOTE Training Samples    : {len(X_train_smote):,}")
print(f"Test Samples              : {len(X_test):,}")

print("\nReady for model training.")