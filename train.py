import torch
import argparse
import sys
import os
import time

from dataset.dataset import dataset
from evaluator.evaluator import evaluator
from models.models import ppi_model
from preprocess.preprocess import df2arr
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import numpy as np
import pandas as pd

def train(model, dataloader, optimizer, evaluators, device, mode='train'):

    epoch_loss = 0
    epoch_count = 1
    concate_predict = np.empty((0, 2), float)
    concate_target = np.empty((0, 1), float)

    for pirna, mrna, label in tqdm(dataloader, ncols=60):
        pirna = torch.tensor(pirna, dtype=torch.float32, device=device)
        mrna = torch.tensor(mrna, dtype=torch.float32, device=device)
        label = torch.tensor(label, dtype=torch.float32, device=device)

        predictions, losses = model(pirna, mrna, label)
        predictions = predictions.detach().cpu().numpy()

        label = np.expand_dims(label.detach().cpu().numpy(), axis=1)
        loss = losses.mean()

        concate_predict = np.append(concate_predict, predictions, axis=0)
        concate_target = np.append(concate_target, label, axis=0)

        batch_count = len(predictions)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = (epoch_loss * epoch_count + loss.item() * batch_count) / (epoch_count +  batch_count)
        epoch_count += batch_count

    confusion_matrix = list(evaluators.confusion_matrix(concate_predict, concate_target))
    mcc = evaluators.mcc(concate_predict, concate_target)
    auc = evaluators.auc(concate_predict, concate_target)

    return round(epoch_loss, 4), confusion_matrix, round(mcc, 4), round(auc, 4)

def calc(c_matrix):

    tn, fp, fn, tp = c_matrix
    acc = (tp + tn) / (tn + fp + fn + tp) * 100.0
    precision = tp / (tp + fp) * 100.0
    recall = tp / (tp + fn) * 100.0
    specificity = tn / (fp + tn) * 100.0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100.0

    return round(acc, 4), round(precision, 4), round(recall, 4), round(specificity, 4), round(f1, 4),

def all_mode(train_path, valid_path, test_path):
    early_patience = 3
    epochs = 100
    lr = 0.001
    batch_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = pd.read_csv(train_path)[['piRNA_seq', 'mRNA_site', 'label']]
    valid_data = pd.read_csv(valid_path)[['piRNA_seq', 'mRNA_site', 'label']]
    test_data = pd.read_csv(test_path)[['piRNA_seq', 'mRNA_site']]

    pi_train, pi_val, pi_test, m_train, m_val, m_test, y_train, y_val = df2arr(train_data, valid_data, test_data)
    # pi_train, pi_test, y_train, y_test = train_test_split(piRNA_x, y, test_size=0.1, random_state=42)
    # m_train, m_test, y_train, y_test = train_test_split(mRNA_x, y, test_size=0.1, random_state=42)

    train_dataset = dataset(pi_train, m_train, y_train)
    val_dataset = dataset(pi_val, m_val, y_val)
    test_dataset = dataset(pi_test, m_test, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = ppi_model()
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.1, patience=4)

    eva = evaluator()

    min_val_loss = 999
    max_val_acc = -999
    best_param = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_c_matrix, train_mcc, train_auc = train(model, train_dataloader, optimizer, eva, device)
        train_acc, train_pre, train_re, train_spec, train_f1 = calc(train_c_matrix)

        model.eval()
        val_loss, val_c_matrix, val_mcc, val_auc = train(model, val_dataloader, None, eva, device, 'val')
        val_acc, val_pre, val_re, val_spec, val_f1 = calc(val_c_matrix)

        # _, test_c_matrix, test_mcc = train(model, test_dataloader, None, eva, device, 'test')
        # test_acc = calc(test_c_matrix)

        print(f'Epoch: {epoch}')
        for param_group in optimizer.param_groups:
            print(f'lr: {round(param_group["lr"], 5)}')
        print(f'train: loss: {train_loss} acc: {train_acc} mcc: {train_mcc}')
        print(f'val: loss: {val_loss} acc: {val_acc} mcc: {val_mcc}')
        # print(f'test: acc: {test_acc} mcc: {test_mcc}')

        scheduler.step(val_loss)

        if max_val_acc < val_acc:
            best_param = model.state_dict()
            info = [epoch, val_loss, val_acc, val_mcc, val_pre, val_re, val_spec, val_auc, val_f1]
            max_val_acc = val_acc
            print(f'Best epoch: {epoch} val_loss: {val_loss} acc: {val_acc} mcc: {val_mcc}\n')
            print(f'val_pre: {val_pre} re: {val_re} spec: {val_spec} auc: {val_auc} f1: {val_f1}\n')
        else:
            early_patience -= 1

        if 0 == early_patience:
            print(f'Early stop at epoch {epoch}')
            print(f'\nBest epoch: {info[0]} val_loss: {info[1]} acc: {info[2]} mcc: {info[3]}')
            print(f'val_pre: {info[4]} re: {info[5]} spec: {info[6]} auc: {info[7]} f1: {info[8]}\n')
            # with open('result.txt', 'a')as f:
            #     result = ','.join([str(v) for v in info[1:]])
            #     f.write(f'{result}\n')
            break
    print('train done')
    model_name = str(time.time())
    torch.save(best_param, f'./ckpt/{model_name}.pt')
    result = testing(best_param, test_dataloader, eva, device)
    test_data['label'] = result
    test_data.to_csv(test_path, index=False)
    print('test done')

def testing(param, dataloader, eva, device):
    test_model = ppi_model(mode='test')
    test_model.load_state_dict(param)
    test_model = test_model.to(device)
    test_model.eval()

    result = []
    for pirna, mrna in tqdm(dataloader, ncols=60):
        pirna = torch.tensor(pirna, dtype=torch.float32, device=device)
        mrna = torch.tensor(mrna, dtype=torch.float32, device=device)

        predictions = test_model(pirna, mrna)
        predictions = predictions.detach().cpu().numpy()
        result.extend(predictions)
    result = eva.preprocess(result)

    return result

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", dest="mode", type=str, choices=["all", "test"], required=True)
    parser.add_argument("-tr", dest="train_path", type=str)
    parser.add_argument("-va", dest="valid_path", type=str)
    parser.add_argument("-te", dest="test_path", type=str)
    parser.add_argument("-model", dest="model_path", type=str)
    args = parser.parse_args()

    if args.mode == "all":
        if not os.path.isfile(args.train_path):
            print(f"{train_path} doesn't exist")
        if not os.path.isfile(args.valid_path):
            print(f"{valid_path} doesn't exist")
        if not os.path.isfile(args.test_path):
            print(f"{test_path} doesn't exist")
        all_mode(args.train_path, args.valid_path, args.test_path)

    if args.mode == "test":
        if not os.path.isfile(args.test_path):
            print(f"{test_path} doesn't exist")
        if not os.path.isfile(args.model_path):
            print(f"{model_path} doesn't exist")

        best_param = torch.load(args.model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        eva = evaluator()

        test_data = pd.read_csv(args.test_path)[['piRNA_seq', 'mRNA_site']]
        pi_test, m_test = df2arr(test_data=test_data, mode='test')
        test_dataset = dataset(pi_test, m_test, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=512)

        result = testing(best_param, test_dataloader, eva, device)
        test_data['label'] = result
        try:
            test_data.to_csv(args.test_path, index=False)
            print('test done')
        except Exception as e:
            print(e)
