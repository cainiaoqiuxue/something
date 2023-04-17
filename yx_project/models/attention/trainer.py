import os
import time
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

def trainer(model, data, epochs=20000):
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data) * epochs)
    loss_func = nn.MSELoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('use device: {}'.format(device))

    loss_threshold = 0.5*1e-9
    loss = 1
    # for epoch in range(epochs):
    epoch = 1
    cost = time.time()
    model = model.to(device)
    model.train()
    while loss > loss_threshold and epoch < epochs:
        for x, y in data:
            x = x.unsqueeze(1)
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_func(prediction, (y - 0.002738) / 0.000760)
            if epoch % 100 == 0:
                # print('{e} / {es}'.format(e=epoch + 1, es=epochs))
                print('Epoch: {: 4d} | Loss: {:.3f} | Cost: {:.2f}'.format(epoch, loss.item(), time.time() - cost))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        epoch += 1

    model.to('cpu')
    model_path = os.path.join(os.path.dirname(__file__), 'attention_model_v3.pkl')
    torch.save(model.state_dict(), model_path)

    return model