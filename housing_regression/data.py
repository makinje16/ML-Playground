import kagglehub
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

feature_columns = [
    "name",
    "total_population",
    "total_households",
    "average_household_income",
    "average_house_age",
    "total_rooms",
    "total_bedrooms",
    "longitude",
    "latitude",
    "ocean_proximity",
]

label_column = 'median_house_value'

class CaliforniaHousingDataset(Dataset):
    def __init__(self, feature_columns, label_column):
        self.features = torch.tensor(feature_columns, dtype=torch.float32)
        self.labels = torch.tensor(label_column.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def GetDataset() -> tuple[CaliforniaHousingDataset, CaliforniaHousingDataset, CaliforniaHousingDataset]:
    # Preprocessing and loading data
    kagglehub.dataset_download("ebelmagnin/housing")
    housingData = pd.read_csv("/home/makinje/.cache/kagglehub/datasets/ebelmagnin/housing/versions/1/california_housing_updated.csv")

    # Label Encode Ocean Proximity
    housingData['ocean_proximity'] = housingData['ocean_proximity'].astype('category')
    housingData['ocean_proximity'] = housingData['ocean_proximity'].cat.codes

    # Label Encode County Name
    housingData['name'] = housingData['name'].astype('category')
    housingData['name'] = housingData['name'].cat.codes

    # Remove Nan values
    housingData[label_column] = housingData[label_column].replace(-666666666.0, np.nan)
    housingData.dropna(subset=[label_column], inplace=True)

    # Split data between train, test, cross validation
    X, y = housingData[feature_columns], housingData[label_column] / 1_000_000

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale Values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        CaliforniaHousingDataset(X_train, y_train), 
        CaliforniaHousingDataset(X_val, y_val),
        CaliforniaHousingDataset(X_test, y_test)
    )