from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset

class DataProcess:
    def __init__(self):
        data_dir = Path(__file__) / "../../data"
        self.train = pd.read_csv(data_dir / 'train_data.csv')
        self.dev = pd.read_csv(data_dir / 'dev_data.csv')
        self.test = pd.read_csv(data_dir / 'test_data.csv')
        self.df = pd.concat([self.train, self.dev, self.test], ignore_index=True)
        # self.data = self.df[self.df['SoH'] != -100].reset_index(drop=True)
        self.sc_map = dict()
        self.sc_cols = ['cycle', 'capacity', 'resistance', 'CCCT', 'CVCT']
        self.cs_map = OneHotEncoder()

    def feature_fillna(self):
        for col in self.df.columns:
            percent = self.df[col].isna().mean()
            if percent > 0:
                print("feature {} has null value: {}".format(col, percent))
                self.df[col].fillna(self.df[col].mean(), inplace=True)

    def feature_sc(self):
        for col in self.sc_cols:
            sc = StandardScaler()
            self.df[col] = sc.fit_transform(self.df[col].values.reshape(-1, 1))
            self.sc_map[col] = sc

    def feature_one_hot(self):
        cs_name = self.cs_map.fit_transform(self.df['CS_Name'].values.reshape(-1, 1)).toarray()
        cs_series = pd.DataFrame(cs_name, columns=self.cs_map.categories_[0])
        self.df = pd.concat([self.df, cs_series], axis=1)

    def get_train_test_data(self):
        feature_cols = ['cycle', 'capacity', 'resistance', 'CCCT', 'CVCT', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
        label_col = 'SoH'

        feature = self.df[self.df['SoH'] != -100][feature_cols].reset_index(drop=True)
        label = self.df[self.df['SoH'] != -100][label_col].reset_index(drop=True)

        train_x, test_x, tran_y, test_y = train_test_split(feature, label, random_state=42, test_size=0.1)
        return train_x, test_x, tran_y, test_y
    
    def forward(self):
        self.feature_fillna()
        self.feature_sc()
        self.feature_one_hot()
        return self.get_train_test_data()
    
    def process(self, data):
        for col in data.columns:
            percent = data[col].isna().mean()
            if percent > 0:
                data[col].fillna(data[col].mean(), inplace=True)

        for col in self.sc_cols:
            sc = self.sc_map[col]
            data[col] = sc.transform(data[col].values.reshape(-1, 1))

        cs_name = self.cs_map.transform(data['CS_Name'].values.reshape(-1, 1)).toarray()
        cs_series = pd.DataFrame(cs_name, columns=self.cs_map.categories_[0])
        data = pd.concat([data, cs_series], axis=1)
        feature_cols = ['cycle', 'capacity', 'resistance', 'CCCT', 'CVCT', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
        return data[feature_cols]



class SoHDataset(Dataset):
    def __init__(self, feature, label):
        super().__init__()
        self.x = feature.values
        self.y = label.values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.float32)
        return x, y