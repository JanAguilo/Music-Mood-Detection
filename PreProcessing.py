import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the datasets
train_path = "/mnt/c/Users/Pol/Desktop/practica2/data4-Bio/TrainSet.csv"
test_path = "/mnt/c/Users/Pol/Desktop/practica2/data4-Bio/TestSet.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Drop filename column (not useful for classification)
train_df.drop(columns=['File_Name'], inplace=True)
test_df.drop(columns=['File_Name'], inplace=True)

# Handle missing values (impute numerical features with median)
numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())

# Encode categorical target variable
label_encoder = LabelEncoder()
train_df['class'] = label_encoder.fit_transform(train_df['class'])

# Normalize numerical features
scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

# Save processed datasets
train_df.to_csv("/mnt/data/Processed_TrainSet.csv", index=False)
test_df.to_csv("/mnt/data/Processed_TestSet.csv", index=False)

print("Preprocessing completed. Data is cleaned and ready for feature selection.")
