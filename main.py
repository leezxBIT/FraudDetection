import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, GraphSAINTSampler, GraphSAINTNodeSampler, RandomNodeSampler
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score
import pdb

from paras import args
from utils import seed_all, load_data, data_preprocess
from models import LINKX, GCN
from plot import *
from loss import *

def train(loader):
    start_time = time.time()
    model.train()
    l = tqdm(loader)
    loss_accum = 0

    for step, batch in enumerate(l):
        batch = batch.to(device)
        batch_train_idx = batch.train_mask.to(torch.bool)
        optimizer.zero_grad()

        pred = model(batch)
        temp_loss = criterion(pred[batch_train_idx], batch.y[batch_train_idx].squeeze())
        # pdb.set_trace()
        loss = temp_loss
        loss_accum += temp_loss.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    print(f'Epoch: {epoch}, loss: {loss.item() / len(loader)}, time: {time.time() - start_time}s')



def eval(data, test):
    # pdb.set_trace()
    model.eval()
    if not test:
        data = data.to(device)
        idx = data.valid_mask.to(torch.bool)
        y_true = data.y[idx].cpu().numpy()
    elif test:
        data = data.to(device)
        idx = data.test_mask.to(torch.bool)
        y_true = data.y[idx].cpu().numpy()

    with torch.no_grad():
        y_preds = torch.sigmoid(model(data))[idx].cpu().numpy()
    
    
    auc_gnn = roc_auc_score(y_true, y_preds[:,1].tolist())
    precision_gnn = precision_score(y_true, y_preds.argmax(axis=1), zero_division=0, average="macro")
    a_p = average_precision_score(y_true, y_preds[:,1].tolist())
    recall_gnn = recall_score(y_true, y_preds.argmax(axis=1), average="macro")
    f1 = f1_score(y_true, y_preds.argmax(axis=1), average="macro")

    return auc_gnn, precision_gnn, a_p, recall_gnn, f1



if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)
    # === Load and preprocess dataset ===
    data = load_data(args.dataset)
    idx_train, idx_valid, idx_test, cls_num_list = data_preprocess(data, args.dataset, args.homo, 
                                                     args.train_ratio, args.test_ratio, device)

    # === Build model ===
    if args.model == 'LINKX':  
        model = LINKX(data.num_node_features, args.hidden_dim, 2, args.num_layers, data.num_nodes).to(device)
    elif args.model == 'gcn':
        model = GCN(data.num_node_features, args.hidden_dim, 2).to(device)
    # === TODO: More model needed ===

    # === Training Loss ===
    if args.LT == 'CE':
        criterion = LogitAdjustLoss()
    elif args.LT == 'LA':
        criterion = LogitAdjustLoss(cls_num_list)
        criterion.to(device)
    elif args.LT == 'RW':
        criterion = ReWeighting(cls_num_list)
        criterion.to(device)
    elif args.LT == 'ALA':
        criterion = ALALoss(cls_num_list)
        criterion.to(device)

    if args.sample == 'neighbor':
        train_loader = NeighborLoader(data, num_neighbors=[15]*2, batch_size = args.batch_size, input_nodes=torch.tensor(idx_train))
        val_loader =  NeighborLoader(data, num_neighbors=[15]*2, batch_size = args.batch_size, input_nodes=torch.tensor(idx_valid))
        test_loader =  NeighborLoader(data, num_neighbors=[15]*2, batch_size = args.batch_size, input_nodes=torch.tensor(idx_test))
    elif args.sample == 'graphsaint-node':
        train_loader = GraphSAINTNodeSampler(data, batch_size=args.batch_size, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
        val_loader = RandomNodeSampler(data, num_parts=args.test_num_parts, shuffle=True, num_workers=0)
        test_loader = RandomNodeSampler(data, num_parts=args.test_num_parts, shuffle=True, num_workers=0)
    elif args.sample == 'graphsaint':
        train_loader = GraphSAINTNodeSampler(data, batch_size=args.batch_size, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
        val_loader = GraphSAINTNodeSampler(data, batch_size=args.batch_size, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
        test_loader = GraphSAINTNodeSampler(data, batch_size=args.batch_size, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # === Train and evaluation ===
    for epoch in range(1, args.epochs+2):
        train(train_loader)
        val_auc, val_precision, val_ap, val_recall, val_f1 = eval(data, False)
        test_auc, test_precision, test_ap, test_recall, test_f1 =eval(data, True)

        print('Val_AUC: {}, Val_Precision: {}, Val_Average Precision: {}, Val_Recall: {}, Val_F1: {}'.format(
            round(val_auc, 4), round(val_precision, 4), round(val_ap, 4), round(val_recall, 4), round(val_f1, 4)
        ))

        print('Test_AUC: {}, Test_Precision: {}, Test_Average Precision: {}, Test_Recall: {}, Test_F1: {}'.format(
            round(test_auc, 4), round(test_precision, 4), round(test_ap, 4), round(test_recall, 4), round(test_f1, 4)
        ))

    plot_Matrix(model, test_loader, device, dataset='yelp', loss=args.LT, AUC=test_auc)

