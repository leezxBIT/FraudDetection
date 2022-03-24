import random
import numpy as np
import torch
from scipy.io import loadmat
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split

def seed_all(seed):

    ''' initial all the seed '''

    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_edge_index(spa_mat):

    '''Get edge index from sparse matrix in tensor format'''

    rows = spa_mat.nonzero()[0]
    cols = spa_mat.nonzero()[1]
    return torch.tensor(np.vstack((rows, cols)), dtype=torch.long)


def load_data(dataset, homo=True): 

    """ load and preprocess dataset """ 
   
    if dataset == 'yelpchi':
        yelp = loadmat('data/YelpChi.mat')
        if homo:
            data = Data()
            data.x = torch.tensor(yelp['features'].toarray(), dtype = torch.float)
            data.y = data.y = torch.tensor(yelp['label'].flatten(),  dtype=torch.long)
            data.edge_index = get_edge_index(yelp['homo'])

        else:
            data = HeteroData()
            data['r'].x = torch.tensor(yelp['features'].toarray(), dtype = torch.float)
            data.y = torch.tensor(yelp['label'].flatten(),  dtype=torch.long)
            data['r','u','r'].edge_index = get_edge_index(yelp['net_rur'])
            data['r','t','r'].edge_index = get_edge_index(yelp['net_rtr'])
            data['r','s','r'].edge_index = get_edge_index(yelp['net_rsr'])
    
    elif dataset == 'amazon':
        amz = loadmat('data/Amazon.mat')
        if homo:
            data = Data()
            data.x = torch.tensor(amz['features'].toarray(), dtype = torch.float)
            data.y = torch.tensor(amz['label'].flatten(),  dtype=torch.long)
            data.edge_index = get_edge_index(amz['homo'])

        else:
            data = HeteroData()
            data['u'].x = torch.tensor(amz['features'].toarray(), dtype = torch.float)
            data.y = torch.tensor(amz['label'].flatten(),  dtype=torch.long)
            data['u','p','u'].edge_index = get_edge_index(amz['net_upu'])
            data['u','s','u'].edge_index = get_edge_index(amz['net_usu'])
            data['u','v','u'].edge_index = get_edge_index(amz['net_uvu'])

    return data

def data_preprocess(data, name, homo, train_ratio, test_ratio, device):

    ''' Split the dataset and normalization '''

    if name == 'yelpchi':
        NormalizeFeatures(data.x if homo else data['r'].x)
        index = list(range(len(data.y)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, data.y, stratify=data.y, train_size=train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=test_ratio,
                                                                random_state=2, shuffle=True)

    elif name == 'amazon':
        # === 0-3304 are unlabeled nodes ===
        NormalizeFeatures(data.x if homo else data['u'].x)
        index = list(range(3305, len(data.y)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, data.y[3305:], stratify=data.y[3305:],
                                                                train_size=train_ratio, random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=test_ratio, random_state=2, shuffle=True)
                                                                
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

    train_mask[idx_train] = True
    valid_mask[idx_valid] = True
    test_mask[idx_test] = True
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    cls_num_list = []
    for i in range(2):
        cls_mask = torch.bitwise_and(data.train_mask.cpu(), (data.y == i))
        cls_num_list.append(cls_mask.sum().item())

    return idx_train, idx_valid, idx_test, cls_num_list
