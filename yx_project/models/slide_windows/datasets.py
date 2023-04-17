import datetime
import torch
from torch.utils.data import Dataset, DataLoader

class AttentionSlideDataSet(Dataset):
    def __init__(self, feature_df, label_df, windows=3):
        self.x = feature_df.values
        self.y = label_df.values.reshape(-1, 1)
        self.windows = windows

        self.x = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)

        self.x = self.slide_feature()


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item, :], self.y[item]

    @staticmethod
    def date_range(date, window):
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
        start = date - datetime.timedelta(days=window - 1)
        res = [start]
        for i in range(1, window):
            res.append(start + datetime.timedelta(days=i))
        return res

    def slide_feature(self):
        features = torch.cat([torch.zeros(self.windows - 1, self.x.size(1)), self.x])
        res = torch.tensor([])
        for i in range(self.windows - 1, self.windows + self.x.size(0) - 1):
            tmp = features[i - self.windows + 1:i + 1, :].reshape(1, -1)
            res = torch.cat([res, tmp])
        return torch.FloatTensor(res)


def get_data(feature_df, label_df, windows, batch_size=None, type=False):
    data = AttentionSlideDataSet(feature_df, label_df, windows=windows)
    if type:
        return data
    if batch_size is None:
        batch_size = feature_df.shape[0]
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader