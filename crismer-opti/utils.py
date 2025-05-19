import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import softmax
import joblib

pwd = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
def save_pkl(data, pkl):
    with open(pkl, 'wb') as f:
        pickle.dump(data, f)
    return 

def load_pkl(pkl):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    return data

def one_hot_features(df):
    nucleotides = ['A', 'T', 'G', 'C']
    pairs = [f'{n1}{n2}' for n1 in nucleotides for n2 in nucleotides]
    pairwise_features = np.zeros((len(df), 20, len(pairs)))
    
    for idx, row in df.iterrows():
        on_seq = row['On']
        off_seq = row['Off']
        
        for pos in range(20):
            pair = on_seq[pos] + off_seq[pos]
            if pair in pairs:
                pair_idx = pairs.index(pair)
                pairwise_features[idx, pos, pair_idx] = 1
    
    return pairwise_features

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MultiBranchConv(nn.Module):
    def __init__(self, output_channels=16, attention=True):
        super(MultiBranchConv, self).__init__()
        self.branch1 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(1, 16))
        self.branch2 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(2, 16))
        self.branch3 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(3, 16))
        self.branch4 = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(4, 16))
        self.attn = attention
        self.ca1 = ChannelAttention(output_channels)
        self.ca2 = ChannelAttention(output_channels)
        self.ca3 = ChannelAttention(output_channels)
        self.ca4 = ChannelAttention(output_channels)
        self.sa1 = SpatialAttention(kernel_size=7)
        self.sa2 = SpatialAttention(kernel_size=7)
        self.sa3 = SpatialAttention(kernel_size=7)
        self.sa4 = SpatialAttention(kernel_size=7)

    def forward(self, x):
        out1 = F.relu(self.branch1(x))
        x_pad2 = F.pad(x, (0, 0, 0, 1))
        out2 = F.relu(self.branch2(x_pad2))
        x_pad3 = F.pad(x, (0, 0, 1, 1))
        out3 = F.relu(self.branch3(x_pad3))
        x_pad4 = F.pad(x, (0, 0, 1, 2))
        out4 = F.relu(self.branch4(x_pad4))

        if self.attn:
            out1 = out1 * self.ca1(out1) * self.sa1(out1)
            out2 = out2 * self.ca2(out2) * self.sa2(out2)
            out3 = out3 * self.ca3(out3) * self.sa3(out3)
            out4 = out4 * self.ca4(out4) * self.sa4(out4)

        out1 = out1.squeeze(-1).transpose(1, 2)
        out2 = out2.squeeze(-1).transpose(1, 2)
        out3 = out3.squeeze(-1).transpose(1, 2)
        out4 = out4.squeeze(-1).transpose(1, 2)
        output = torch.cat((out1, out2, out3, out4), dim=-1)
        
        return output

class CRISPRTransformerModel(nn.Module):
    def __init__(self, config):
        super(CRISPRTransformerModel, self).__init__()
        self.input_dim = 64
        self.num_layers = config.get("num_layers", 2)
        self.num_heads = config.get("num_heads", 8)
        self.dropout_prob = config["dropout_prob"]
        self.number_hidden_layers = config["number_hidder_layers"]
        self.seq_length = config.get("seq_length", 20)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_length, self.input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            dim_feedforward=self.input_dim * 4,
            dropout=self.dropout_prob,
            batch_first=True,
            norm_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        self.conv = MultiBranchConv(attention=config["attn"])
        self.hidden_layers = []
        start_size = self.seq_length * self.input_dim
        for i in range(self.number_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(start_size, start_size // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_prob)
            )
            self.hidden_layers.append(layer)
            start_size = start_size // 2
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(start_size, 2)

    def forward(self, x, src_mask=None):
        x = self.conv(x)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return x

class TrainerDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def tester(model, test_x, test_y):
    test_dataset = TrainerDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    results = []
    true_labels = []
    with torch.no_grad():
        for test_features, test_labels in test_dataloader:
            outputs = model(test_features.to(device)).detach().to("cpu")
            results.extend(outputs)
            true_labels.extend(test_labels)
    return true_labels, results

class Stats:
    def __init__(self):
        self.acc = 0
        self.pre = 0
        self.re = 0
        self.f1 = 0
        self.roc = 0
        self.prc = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.scores = []
        self.results = []

    def print(self):
        print('Accuracy: %.4f' % self.acc)
        print('Precision: %.4f' % self.pre)
        print('Recall: %.4f' % self.re)
        print('F1 Score: %.4f' % self.f1)
        print('ROC: %.4f' % self.roc)
        print('PR AUC: %.4f' % self.prc)
        print("Confusion Matrix")
        print(self.tn, "\t", self.fp)
        print(self.fn, "\t", self.tp)

def eval_matrices(model, test_x, test_y, debug=True, scaler="minmax"):
    true_y, results = tester(model, test_x, test_y)
    predictions = [torch.nn.functional.softmax(r, dim=0) for r in results]
    pred_y = np.array([y[1].item() for y in predictions])
    test_y = np.array([y.item() for y in true_y])
    raw_logits = np.array([r[1].item() for r in results])

    if scaler == "minmax":
        scaled_scores = MinMaxScaler().fit_transform(raw_logits.reshape(-1, 1)).flatten()
    elif scaler == "standard":
        scaled_scores = StandardScaler().fit_transform(raw_logits.reshape(-1, 1)).flatten()
    else:
        scaled_scores = raw_logits

    threshold = 0.5
    pred_y_list = (pred_y > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y_list).ravel()
    precision, recall, _ = precision_recall_curve(test_y, pred_y)
    auc_score = auc(recall, precision)
    acc = accuracy_score(test_y, pred_y_list)
    pr = tp / (tp + fp) if (tp + fp) > 0 else -1
    re = tp / (tp + fn) if (tp + fn) > 0 else -1
    f1 = (2 * pr * re / (pr + re)) if (pr + re) > 0 else -1

    stats = Stats()
    stats.acc = acc
    stats.pre = pr
    stats.re = re
    stats.f1 = f1
    stats.roc = roc_auc_score(test_y, raw_logits)
    stats.prc = auc_score
    stats.tn = tn
    stats.fp = fp
    stats.fn = fn
    stats.tp = tp
    stats.scores = scaled_scores
    stats.results = results

    if debug:
        stats.print()

    return stats

class PredictDataset(Dataset):
    def __init__(self, inputs):
        # Match preprocessing from TrainerDataset
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]  # Only return features

def predictor(model, test_x):
    test_dataset = PredictDataset(test_x)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()
    results = []
    with torch.no_grad():
        for batch_features in test_dataloader:
            outputs = model(batch_features.to(device))
            results.extend(outputs.detach().cpu())
    return results

def get_score_without_label(model, test_x, scaler, T=10):
    results = predictor(model, test_x)
    

    predictions = [torch.nn.functional.softmax(r/T, dim=0) for r in results]
    scaled_scores = np.array([y[1].item() for y in predictions])
        # scaler_instance = MinMaxScaler()
    scaled_scores = scaler.transform(scaled_scores.reshape(-1, 1)).flatten()
        # joblib.dump(scaler_instance, "minmax_scaler.pkl")  # Save the scaler

    

    return scaled_scores
