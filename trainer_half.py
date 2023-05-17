import torch
from torch.cuda.amp import autocast as autocast

def train(train_loader, model, criterion, sensing_rate, optimizer, device, scaler):
    model.train()
    sum_loss = 0
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        # outputs,x2,x3,sys_loss = model(inputs)
        gamma1 = torch.Tensor([0.01]).to(device)
        I = torch.eye(int(sensing_rate * 1024)).to(device)
        with autocast():
            outputs, sys_loss = model(inputs)
            assert torch.isnan(sys_loss).sum() == 0, print(sys_loss)
            assert torch.isnan(outputs).sum() == 0, print(outputs)
            loss = criterion(outputs, inputs) + torch.mul(criterion(sys_loss, I), gamma1)
            assert torch.isnan(loss).sum() == 0, print(loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sum_loss += loss.item()
    return sum_loss
