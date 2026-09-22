import numpy as np
import torch
import torch.nn as nn
import time

SEQ_LEN = 24
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 64

torch.manual_seed(42)
np.random.seed(42)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SmallTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


def make_windows(series: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def fit_predict_transformer(train_series, test_series, seq_len=SEQ_LEN):
    mean, std = train_series.mean(), train_series.std() + 1e-8
    train_norm = (train_series.values - mean) / std
    test_norm = (test_series.values - mean) / std

    X_train, y_train = make_windows(train_norm, seq_len)
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    model = SmallTransformer()
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