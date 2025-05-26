import pandas as pd
from sklearn.model_selection import train_test_split
import os


# Load the Excel file
data = pd.read_excel("dataset/adult_income_male/adult_dataset_male.xlsx")

# Split data into train (70%), temp (30%)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)

# Split temp data into validation (50%) and test (50%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the datasetsdataset/adult_income_male/adult_dataset_male.xlsx
train_data.to_csv("train_data.csv", index=False)
val_data.to_csv("val_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("Data partitioned successfully!")
