# 导入必要的库
import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# 定义一个神经网络模型类
class SoHDNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers) -> None:
        super().__init__()  # 调用父类的初始化方法
        # 定义模型的输入维度、输出维度、隐藏层维度和层数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        # 定义一个多层感知器（MLP）作为模型的第一部分
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),  # 线性层
            nn.LayerNorm(self.hidden_dim),  # 层归一化
            nn.ReLU()  # 激活函数
        )

        # 定义前馈网络（FFN），用于模型的后续层
        self.ffn = [
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 线性层
            nn.LayerNorm(self.hidden_dim),  # 层归一化
            nn.ReLU()  # 激活函数
        ] * self.layers  # 根据层数复制FFN
        self.ffn = nn.Sequential(*self.ffn)  # 将FFN层合并为一个Sequential模块

        # 定义模型的输出层
        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # 前向传播函数
        out = self.mlp(x)  # 通过MLP层
        if self.layers > 3:  # 如果层数大于3，则使用残差连接
            out = self.ffn(out) + out
        else:
            out = self.ffn(out)
        out = self.head(out)  # 通过输出层
        return out

# 定义训练函数
def train(model, 
          train_dataset, 
          epochs=5,
          log_step=1,
          valid_dataset=None,
          save_dir = None
          ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择设备
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 创建训练数据加载器
    no_decay = ['bias', 'LayerNorm.weight']  # 定义不进行权重衰减的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.05},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)  # 创建优化器
    loss_func = nn.MSELoss()  # 定义损失函数

    begin_time = time.time()  # 记录开始时间
    for epoch in range(epochs):
        print("Epoch: {} / {} (device: {})".format(epoch + 1, epochs, device))
        # 训练过程
        model.train()  # 设置模型为训练模式
        for idx, batch in enumerate(train_dataloader):
            output = model(batch[0])  # 得到模型输出
            loss = loss_func(output, batch[1].reshape(-1, 1))  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if idx % log_step == 0:
                print("|step: {:3d} |loss: {:.3f} |".format(idx, loss.item()))  # 打印日志

        # 验证过程
        if valid_dataset is not None:
            eval_labels = []
            eval_predictions = []
            valid_dataloader = DataLoader(valid_dataset, batch_size=64)  # 创建验证数据加载器
            model.eval()  # 设置模型为评估模式

            for idx, batch in enumerate(valid_dataloader):
                with torch.no_grad():  # 不计算梯度
                    output = model(batch[0])
                eval_labels.append(batch[1])
                eval_predictions.append(output)
            labels = torch.cat(eval_labels)  # 合并标签
            predictions = torch.cat(eval_predictions)  # 合并预测结果
            mse = mean_squared_error(labels.numpy(), predictions.numpy())  # 计算均方误差
            rmse = root_mean_squared_error(labels.numpy(), predictions.numpy())  # 计算均方根误差
            print('|valid dataset: |MAE: {} |RMSE: {}'.format(mse, rmse))  # 打印验证结果

        # 保存模型
        if save_dir is not None:
            now_time = time.localtime()  # 获取当前时间
            model_name = '{}_{}_{}_{}_{}_epoch_{}.pth'.format('model',
                                                          now_time.tm_mon,
                                                          now_time.tm_mday,
                                                          now_time.tm_hour,
                                                          now_time.tm_min,
                                                          epoch + 1)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))  # 保存模型参数