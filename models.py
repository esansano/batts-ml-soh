import os.path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class FCNModel(nn.Module):

    def __init__(self, dataset):
        super(FCNModel, self).__init__()
        if dataset == 5:
            self.fc1 = nn.Linear(252, 256)
        elif dataset == 10:
            self.fc1 = nn.Linear(127, 256)
        elif dataset == 20:
            self.fc1 = nn.Linear(64, 256)
        elif dataset == 50:
            self.fc1 = nn.Linear(27, 256)
        else:
            raise ValueError("Dataset not supported")
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        return x


class CNNModel(nn.Module):

    def __init__(self, dataset, c_rate=True):
        super(CNNModel, self).__init__()
        self.c_rate = c_rate
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5)
        self.mpl1 = nn.MaxPool1d(kernel_size=2)
        self.cnn3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.cnn4 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=3)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.50)
        if dataset == 5:
            self.fc1 = nn.Linear(3713, 64)
        elif dataset == 10:
            self.fc1 = nn.Linear(1729, 64)
        elif dataset == 20:
            self.fc1 = nn.Linear(705, 64)
        elif dataset == 50:
            self.fc1 = nn.Linear(129, 64)
        else:
            raise ValueError("Dataset not supported")
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x[:, None, :]
        if self.c_rate:
            x_class = x[:, :, -1]                   # Take the last feature (categorical feature)
            x = x[:, :, :-1]                        # Take all features but the last (use them for regression)

            x = nn.functional.relu(self.cnn1(x))
            x = nn.functional.relu(self.cnn2(x))
            x = self.mpl1(x)
            x = nn.functional.relu(self.cnn3(x))
            x = nn.functional.relu(self.cnn4(x))

            x = self.flat(x)
            x = self.drop(x)
            x = torch.cat((x, x_class), dim=1)      # Concatenates cnn output with x_class
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = nn.functional.relu(self.cnn1(x))
            x = nn.functional.relu(self.cnn2(x))
            x = self.mpl1(x)
            x = nn.functional.relu(self.cnn3(x))
            x = nn.functional.relu(self.cnn4(x))

            x = self.flat(x)
            x = self.drop(x)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)

        return x


def train(model, train_ds, val_ds, test_ds, epochs, batch_size, optimizer, criterion, patience, device,
          dataset, length, loss_criterion, fold, snapshots=False):
    best_epoch_loss = np.inf
    best_epoch = 0
    filename = f'CNN_d{dataset}_l{length[1]}_{loss_criterion}_{fold}'
    model.train()
    pbar = tqdm(range(epochs), leave=False, ncols=100, bar_format='{desc}{rate_fmt}{postfix}')
    x = train_ds.x.to(device)
    y = train_ds.y.to(device)
    for epoch in pbar:
        epoch_loss = 0
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.shape[0]
        epoch_loss = epoch_loss / x.shape[0]

        if type(criterion) == nn.MSELoss:
            epoch_loss = np.sqrt(epoch_loss)

        rmse_val, mae_val = test(model, val_ds.x, val_ds.y, device)

        if loss_criterion == 'RMSE' and rmse_val < best_epoch_loss:
            best_epoch_loss = rmse_val
            best_epoch = epoch
            # save_model(model, os.path.join('snapshots', f'{filename}.pt'))
            save_preds = True
        elif loss_criterion == 'MAE' and mae_val < best_epoch_loss:
            best_epoch_loss = mae_val
            best_epoch = epoch
            # save_model(model, os.path.join('snapshots', f'{filename}.pt'))
            save_preds = True
        else:
            save_preds = False
            if epoch - best_epoch > patience:
                # print(f'Early stopping at epoch {epoch}')
                break
        if snapshots and save_preds:
            test(model, test_ds.x, test_ds.y, device, save=True, filename=f'{filename}.csv')

        desc = f'epoch: {epoch + 1:5d} [{epoch - best_epoch:4d}] - train loss: {epoch_loss:.6f}'
        desc += f' - [{rmse_val:.6f} / {mae_val:.6f}]'
        pbar.set_description(desc)
    return best_epoch_loss, best_epoch


def validate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        y_hat = model(x.to(device))
        se_test = torch.pow(y_hat - y.to(device), 2).detach().cpu().numpy()
        ae_test = torch.abs(y_hat - y.to(device)).detach().cpu().numpy()
    rmse = np.sqrt(np.mean(se_test))
    mae = np.mean(ae_test)
    return rmse, mae


def test(model, x, y, device, save=False, filename=None):
    model.eval()
    with torch.no_grad():
        y_hat = model(x.to(device))
        se_test = torch.pow(y_hat - y.to(device), 2).detach().cpu().numpy()
        ae_test = torch.abs(y_hat - y.to(device)).detach().cpu().numpy()
    rmse = np.sqrt(np.mean(se_test))
    mae = np.mean(ae_test)
    if save:
        path = f'./snapshots/{filename}'
        np.savetxt(path, np.concatenate((y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()), axis=1),
                   delimiter=',')
    return rmse, mae


def save_model(model, path):
    torch.save(model.state_dict(), path)
