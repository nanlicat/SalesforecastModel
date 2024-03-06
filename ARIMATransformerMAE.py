import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

torch.manual_seed(0)
np.random.seed(0)

input_window = 20
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
def create_inout_sequences(input_data,lable_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        # train_label = input_data[i + output_window:i + tw + output_window]
        train_label = lable_data[i :i + tw ]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)
def get_data():
    # MAE:  0.03288857266306877     # pmae:  2.0      # MAE(%):  1.6444286331534386
    originData = pd.read_csv('./merged_data3576yj.csv', usecols=['Value'])
    # originData = pd.read_csv('../Data/merged_data3003rl.csv', usecols=['Value'])
    # originData = pd.read_csv('../Data/merged_30031j.csv', usecols=['Value'])
    # originData = pd.read_csv('../Data/merged_data4238yj.csv', usecols=['Value'])
    model = ARIMA(originData['Value'], order=(1, 1, 0))
    model_fit = model.fit(disp=0)

    # 得到预测集
    # predict =model_fit.forecast(steps=len(test_data))[0]
    series = model_fit.predict(start=1, end=len(originData), typ='levels')
    # series = pd.read_csv('./daily-min-temperatures.csv', usecols=['Temp'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series=series.values.reshape(-1, 1)
    # series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
    series = scaler.fit_transform(series).reshape(-1)

    originData = originData.values.reshape(-1, 1)
    originData = scaler.fit_transform(originData).reshape(-1)

    train_samples = int(0.9 * len(series))
    train_data = series[:train_samples]
    test_data = series
    train_samples_lable = int(0.9 * len(originData))
    train_data_lable = originData[:train_samples_lable]
    test_data_lable = originData
    # train_sequence  (350,2,20)
    train_sequence = create_inout_sequences(train_data,train_data_lable, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data,test_data_lable, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device),scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target
def train(train_data):
    model.train()

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size )
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    # total_mae=0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            # total_mae += mean_absolute_error(output[-1].view(-1).cpu(), target[-1].view(-1).cpu()).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)


    mae = mean_absolute_error(truth, test_result)
    rmse = np.sqrt(mean_squared_error(truth, test_result))
    maeper = mean_absolute_error(truth, test_result) / np.ptp(truth) * 100
    rmseper = np.sqrt(mean_squared_error(truth, test_result)) / np.ptp(truth) * 100
    print('MAE: %.3f' % mae)
    print('RMSE: %.3f' % rmse)
    print('MAE(百分比): %.3f' % maeper)
    print('RMSE(百分比): %.3f' % rmseper)

    plt.plot(test_result, color="red", label="预测值")
    plt.plot(truth, color="blue", label="真实值")

    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend(fontsize=12)  # 添加图例
    plt.rcParams['axes.unicode_minus'] = False
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout(pad=1)
    plt.savefig('ARIMATransformer/transformer-epoch%d.png' % epoch,dpi=400)

    plt.close()

    return total_loss / i

train_data, val_data,scaler = get_data()
model = TransAm().to(device)
criterion = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
epochs = 300

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % 10 is 0):
        val_loss = plot_and_loss(model, val_data, epoch)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)
    scheduler.step()