from config import args
import joblib
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch
import random
from tqdm import tqdm


class MyDataLoader(object):

    def __init__(self, dataset, batch_size, mini_batch_size=0):
        self.total = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        if mini_batch_size == 0:
            self.mini_batch_size = self.batch_size

    def __getitem__(self, item):
        ceil = (item + 1) * self.batch_size
        sub_dataset = self.dataset[ceil - self.batch_size:ceil]
        if ceil >= self.total:
            random.shuffle(self.dataset)
        return DataLoader(sub_dataset, batch_size=self.mini_batch_size)

    def __len__(self):
        if self.total == 0: return 0
        return (self.total - 1) // self.batch_size + 1

def split_train_valid_test(data, train_size, valid_part=0.1):
    train_data = data[:train_size]
    test_data = data[train_size:]
    random.shuffle(train_data)
    valid_size = round(valid_part * train_size)
    valid_data = train_data[:valid_size]
    train_data = train_data[valid_size:]
    return train_data, test_data, valid_data


def get_data_loader(dataset, batch_size, mini_batch_size):
    # param
    train_size = args[dataset]["train_size"]

    # load data
    inputs = np.load(f"temp/{dataset}.inputs.npy")
    input_sentence=np.load(f"temp/{dataset}.inputs_sentence.npy",allow_pickle=True)
    graphs = np.load(f"temp/{dataset}.graphs.npy")
    weights = np.load(f"temp/{dataset}.weights.npy")
    targets = np.load(f"temp/{dataset}.targets.npy")
    len_inputs = joblib.load(f"temp/{dataset}.len.inputs.pkl")
    len_inputs_sentence = joblib.load(f"temp/{dataset}.len.inputs_sentence.pkl")
    len_graphs = joblib.load(f"temp/{dataset}.len.graphs.pkl")
    word2vec = np.load(f"temp/{dataset}.word2vec.npy")


    # py graph dtype
    data = []
    for x,x_sentence, edge_index, edge_attr, y, lx_s,lx, le  in tqdm(zip(
            inputs,input_sentence, graphs, weights, targets,len_inputs_sentence, len_inputs, len_graphs)):

        x = torch.tensor(x[:lx], dtype=torch.long)
        x_sentence=torch.tensor(x_sentence[:lx_s], dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor([e[:le] for e in edge_index], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr[:le], dtype=torch.float)
        lens = torch.tensor(lx, dtype=torch.long)
        lens_s=torch.tensor(lx_s, dtype=torch.long)


        data.append(Data(x=x, x_s=x_sentence,y=y, edge_attr=edge_attr, edge_index=edge_index, length=lens,len_inputs=lens_s))



    # split
    train_data, test_data, valid_data = split_train_valid_test(data, train_size, valid_part=0.1)
    print(len(test_data))
    return [MyDataLoader(data, batch_size=batch_size, mini_batch_size=mini_batch_size)
            for data in [train_data, test_data, valid_data]], word2vec


