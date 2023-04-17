import torch
from torch.utils.data import Dataset, DataLoader

class AttentionDataSet(Dataset):
    def __init__(self, feature_df, label_df):
        self.x = feature_df.values
        self.y = label_df.values.reshape(-1, 1)

        self.x = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item, :], self.y[item]


def get_data(feature_df, label_df, batch_size=None):
    data = AttentionDataSet(feature_df, label_df)
    if batch_size is None:
        batch_size = feature_df.shape[0]
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader