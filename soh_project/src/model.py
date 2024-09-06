import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, root_mean_squared_error


class SoHDNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()

        )
        self.ffn = [
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        ] * self.layers
        self.ffn = nn.Sequential(*self.ffn)

        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        out = self.mlp(x)
        if self.layers > 3:
            out = self.ffn(out) + out
        else:
            out = self.ffn(out)
        out = self.head(out)
        return out
    

def train(model, 
          train_dataset, 
          epochs=5,
          log_step=1,
          valid_dataset=None,
          save_dir = None
          ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.05},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    loss_func = nn.MSELoss()

    begin_time = time.time()
    for epoch in range(epochs):
        print("Epoch: {} / {} (device: {})".format(epoch + 1, epochs, device))
        # train
        model.train()
        for idx, batch in enumerate(train_dataloader):
            output = model(batch[0])
            loss = loss_func(output, batch[1].reshape(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % log_step == 0:
                print("|step: {:3d} |loss: {:.3f} |".format(idx, loss.item(),))
        # eval
        if valid_dataset is not None:
            eval_labels = []
            eval_predictions = []
            valid_dataloader = DataLoader(valid_dataset, batch_size=64)
            model.eval()

            for idx, batch in enumerate(valid_dataloader):
                with torch.no_grad():
                    output = model(batch[0])
                eval_labels.append(batch[1])
                eval_predictions.append(output)
            labels = torch.cat(eval_labels)
            predictions = torch.cat(eval_predictions)
            mse = mean_squared_error(labels.numpy(), predictions.numpy())
            rmse = root_mean_squared_error(labels.numpy(), predictions.numpy())
            print('|valid dataset: |MAE: {} |RMSE: {}'.format(mse, rmse))


        # save
        if save_dir is not None:
            now_time = time.localtime()
            model_name = '{}_{}_{}_{}_{}_epoch_{}.pth'.format('model',
                                                          now_time.tm_mon,
                                                          now_time.tm_mday,
                                                          now_time.tm_hour,
                                                          now_time.tm_min,
                                                          epoch + 1)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))

