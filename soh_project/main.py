import torch
from src.data_process import DataProcess, SoHDataset
from src.model import SoHDNNModel, train


def main():
    dp = DataProcess()
    train_x, test_x, train_y, test_y = dp.forward()

    train_data = SoHDataset(train_x, train_y)
    test_data = SoHDataset(test_x, test_y)
    model = SoHDNNModel(train_x.shape[1], 1, 128, 2)

    train(model, train_data, 5, 10, test_data, save_dir=None)

    result = model(torch.tensor(dp.process(dp.test).values, dtype=torch.float32))
    result = result.detach().numpy().reshape(-1)
    submission = dp.test.copy()
    submission['result'] = result

    idxs = submission[submission['SoH'] != -100].index
    fill_value = submission.loc[idxs, 'SoH'].to_list()
    submission.loc[idxs, 'result'] = fill_value
    submission.to_csv('data/cv.csv', index=False)
    submission = submission[['cycle', 'CS_Name', 'result']]
    submission.to_csv('data/submission.csv')

if __name__ == '__main__':
    main()