import numpy as np
import torch
import torch.nn as nn
import time

SEQ_LEN = 24          # 4 hours of history at 10-min resolution
HIDDEN_SIZE = 32
EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 64

torch.manual_seed(42)
np.random.seed(42)


class SmallLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_windows(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def fit_predict_lstm(train_series, test_series, seq_len=SEQ_LEN):
    # normalize using train statistics only
    mean, std = train_series.mean(), train_series.std() + 1e-8
    train_norm = (train_series.values - mean) / std
    test_norm = (test_series.values - mean) / std

    X_train, y_train = make_windows(train_norm, seq_len)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    model = SmallLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    model.train()
    n = len(X_train_t)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    # rolling one-step-ahead prediction over the test period, using true
    # observed values to build each next window (proper one-step-ahead)
    t0 = time.time()
    model.eval()
    full_norm = np.concatenate([train_norm[-seq_len:], test_norm])
    preds_norm = []
    with torch.no_grad():
        for i in range(len(test_norm)):
            window = full_norm[i:i+seq_len].reshape(1, seq_len, 1).astype(np.float32)
            pred = model(torch.from_numpy(window)).item()
            preds_norm.append(pred)
    inference_time = time.time() - t0

    preds = np.array(preds_norm) * std + mean
    return preds, train_time, inference_time