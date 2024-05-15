import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from code.preprocessing import data_preprocess_v1


def mask(df, p=0.5):
    mask_array = np.random.choice([np.NaN, 1], size=df.shape, p=[p, 1 - p])
    mask_df = df * mask_array
    mask_df = mask_df.fillna(-1)
    return mask_df


class AEdataset(Dataset):
    def __init__(self, df):
        self.df = df.fillna(-1)
        self.train_feature = [f'n{i}' for i in range(1, 15)]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        sample = self.df.iloc[item, :]
        label = sample[self.train_feature]
        sample[self.train_feature] = mask(sample[self.train_feature])
        return torch.tensor(sample,dtype=torch.float), torch.tensor(label,dtype=torch.float)


class AutoEncoder:
    def __init__(self):
        self.ms = MinMaxScaler()
        self.train_feature = [f'n{i}' for i in range(1, 15)]


    def get_data(self):
        feature, _ = data_preprocess_v1()
        drop_index = feature[self.train_feature].isna().any(axis=1)
        feature = feature.drop(feature.index[drop_index]).reset_index(drop=True)
        columns = feature.columns
        feature = self.ms.fit_transform(feature)
        feature = pd.DataFrame(feature, columns=columns)
        feature.fillna(-1, inplace=True)
        return feature

    def make_train_data(self):
        df = self.get_data()
        test_size = df.shape[0] // 10 * 2
        train_df = df.iloc[test_size:, :]
        test_df = df.iloc[:test_size, :]
        train_dataset = AEdataset(train_df)
        test_dataset = AEdataset(test_df)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=512)
        return train_dataloader, test_dataloader

    def ae_model(self, input_dim, hidden_dim, output_dim):
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

        return model

    def train(self, epochs, save_model=False):
        train_data, test_data = self.make_train_data()
        model = self.ae_model(163, 20, 14)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.9)
        for i in range(epochs):
            print('epoch:', i + 1)

            train_loss = 0
            valid_loss = 0
            train_count = 0
            valid_count = 0
            for x, y in train_data:
                pred = model(x)
                loss = loss_func(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_count += 1
            for x, y in test_data:
                pred = model(x)
                loss = loss_func(pred, y)
                valid_count += 1
                valid_loss += loss.item()
            print(f'loss:{train_loss / train_count}  val_loss:{valid_loss / valid_count}')

        if save_model:
            torch.save(model.state_dict(),'../save_model/torchModel.pth')

if __name__=='__main__':
    ae=AutoEncoder()
    ae.train(15)
