import numpy as np
import pandas as pd
import os

import torch
import anndata
import scanpy as sc

import scanpy as sc
import scvelo as scv
import anndata
import csv

from tqdm import tqdm
from numpy.random import choice

from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Dataloader by pytorch
class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    ### Return the length of dataset, number of samples
    def __len__(self):
        return self.y.shape[0]

    ### Get the data and label of corresponding index
    def __getitem__(self, idx):
        # print(idx)
        return self.x[idx], self.y[idx]

def preprocess(counts1, counts2, meta1, meta2):
    cts1 = meta1['cellType'].tolist()
    cts2 = meta2['cellType'].tolist()

    ## common cell types
    common_cts = list(set(cts1).intersection(set(cts2)))
    k = len(common_cts)

    ## New datasets under common cell types
    meta1 = meta1.loc[meta1['cellType'].isin(common_cts)]
    meta2 = meta2.loc[meta2['cellType'].isin(common_cts)]

    cells1 = meta1.index
    cells2 = meta2.index

    counts1 = counts1.loc[cells1, ]
    counts2 = counts2.loc[cells2, ]

    cts1 = meta1['cellType'].tolist()
    cts2 = meta2['cellType'].tolist()

    counts1.index = cts1
    counts2.index = cts2

    return counts1, counts2, k


#### NEEDED FILES
# 1. GeneLength.txt
def counts2FPKM(counts, genelen):
    genelen = pd.read_csv(genelen, sep=',')
    genelen['TranscriptLength'] = genelen['Transcript end (bp)'] - genelen['Transcript start (bp)']
    genelen = genelen[['Gene name', 'TranscriptLength']]
    genelen = genelen.groupby('Gene name').max()
    # intersection
    inter = counts.columns.intersection(genelen.index)
    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values
    # transformation
    totalreads = counts.sum(axis=1)
    counts = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    counts = pd.DataFrame(counts, columns=inter, index=samplename)
    return counts


def FPKM2TPM(fpkm):
    genename = fpkm.columns
    samplename = fpkm.index
    fpkm = fpkm.values
    total = fpkm.sum(axis=1).reshape(-1, 1)
    fpkm = fpkm * 1e6 / total
    fpkm = pd.DataFrame(fpkm, columns=genename, index=samplename)
    return fpkm


def counts2TPM(counts, genelen):
    fpkm = counts2FPKM(counts, genelen)
    tpm = FPKM2TPM(fpkm)
    return tpm

def counts2TMM(counts):

    # intersection
    inter = counts.columns.intersection(genelen.index)
    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values
    # transformation
    totalreads = counts.sum(axis=1)
    counts = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    counts = pd.DataFrame(counts, columns=inter, index=samplename)
    return counts

def ProcessInputData(train_x, test_x, sep=None, datatype='counts', variance_threshold=0.98,
                     scaler="mms",
                     genelenfile=None):

    ### transform to datatype
    if datatype == 'FPKM':
        if genelenfile is None:
            raise Exception("Please add gene length file!")
        print('Transforming to FPKM')
        train_x = counts2FPKM(train_x, genelenfile)
    elif datatype == 'TPM':
        if genelenfile is None:
            raise Exception("Please add gene length file!")
        print('Transforming to TPM')
        train_x = counts2TPM(train_x, genelenfile)
    elif datatype == 'counts':
        print('Using counts data to train model')

    ### variance cutoff
    print('Cutting variance...')
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_threshold)]
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]

    var_cutoff = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_threshold)]
    test_x = test_x.loc[:, test_x.var(axis=0) > var_cutoff]

    ### find intersected genes
    print('Finding intersected genes...')
    inter = train_x.columns.intersection(test_x.columns)
    train_x = train_x[inter]
    test_x = test_x[inter]

    # print('Intersected gene number is ', len(inter))
    ### MinMax process
    print('Scaling...')
    train_x = np.log(train_x + 1)
    test_x = np.log(test_x + 1)

    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x)
        ss_test_x = ss.fit_transform(test_x)

        return ss_train_x, ss_test_x

    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x)
        mms_test_x = mms.fit_transform(test_x)

        return mms_train_x, mms_test_x, inter


# Simulate data
def generate_simulated_data(sc_data, outname=None,
                            d_prior=None,
                            n=500, samplenum=5000,
                            random_state=None, sparse=True, sparse_prob=0.5,
                            rare=False, rare_percentage=0.4):
    # sc_data should be a cell*gene matrix, no null value, txt file, sep='\t'
    # index should be cell names
    # columns should be gene labels
    print('Reading single-cell dataset, this may take 1 min')
    if '.txt' in sc_data:
        sc_data = pd.read_csv(sc_data, index_col=0, sep='\t')
        sc_data.dropna(inplace=True)
        sc_data['celltype'] = sc_data.index
        sc_data.index = range(len(sc_data))
    elif type(sc_data) is pd.DataFrame:
        sc_data.dropna(inplace=True)
        sc_data['celltype'] = sc_data.index
        sc_data.index = range(len(sc_data))
    elif '.h5ad' in sc_data:
        print('You are using H5AD format data, please make sure "CellType" occurs in the adata.obs')
        sc_data = anndata.read_h5ad(sc_data)
        if isinstance(sc_data.X, np.ndarray):
            pass
        else:
            sc_data.X = sc_data.X.toarray()

        sc_data = pd.DataFrame(sc_data.X, index=sc_data.obs["CellType"], columns=sc_data.var.index)
        sc_data.dropna(inplace=True)
        sc_data['celltype'] = sc_data.index
        sc_data.index = range(len(sc_data))

    elif isinstance(sc_data, anndata.AnnData):
        print('You are using H5AD format data, please make sure "CellType" occurs in the adata.obs')
        if isinstance(sc_data.X, np.ndarray):
            pass
        else:
            sc_data.X = sc_data.X.toarray()

        sc_data = pd.DataFrame(sc_data.X, index=sc_data.obs["CellType"], columns=sc_data.var.index)
        sc_data.dropna(inplace=True)
        sc_data['celltype'] = sc_data.index
        sc_data.index = range(len(sc_data))
    else:
        raise Exception("Please check the format of single-cell data!")
    print('Reading dataset is done')

    num_celltype = len(sc_data['celltype'].value_counts())
    genename = sc_data.columns[:-1]

    celltype_groups = sc_data.groupby('celltype').groups
    sc_data.drop(columns='celltype', inplace=True)

    ### normalize with scanpy
    print('Normalizing raw single cell data with scanpy.pp.normalize_total')
    sc_data = anndata.AnnData(sc_data)
    sc.pp.normalize_per_cell(sc_data)
    # sc.pp.normalize_total(sc_data, target_sum=1e4)

    # use ndarray to accelerate
    # change to C_CONTIGUOUS, 10x faster
    sc_data = sc_data.X
    sc_data = np.ascontiguousarray(sc_data, dtype=np.float32)
    # make random cell proportions

    if random_state is not None and isinstance(random_state, int):
        print('You specified a random state, which will improve the reproducibility.')

    if d_prior is None:
        print('Generating cell fractions using Dirichlet distribution without prior info (actually random)')
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(np.ones(num_celltype), samplenum)
        print('RANDOM cell fractions is generated')
    elif d_prior is not None:
        print('Using prior info to generate cell fractions in Dirichlet distribution')
        assert len(d_prior) == num_celltype, 'dirichlet prior is a vector, its length should equals ' \
                                             'to the number of cell types'
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(d_prior, samplenum)
        print('Dirichlet cell fractions is generated')

    # make the dictionary
    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)

    prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    # sparse cell fractions
    if sparse:
        print("You set sparse as True, some cell's fraction will be zero, the probability is", sparse_prob)
        ## Only partial simulated data is composed of sparse celltype distribution
        for i in range(int(prop.shape[0] * sparse_prob)):
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    if rare:
        print(
            'You will set some cell type fractions are very small (<3%), '
            'these celltype is randomly chosen by percentage you set before.')
        ## choose celltype
        np.random.seed(0)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf

    # precise number for each celltype
    cell_num = np.floor(n * prop)

    # precise proportion based on cell_num
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)

    # start sampling
    sample = np.zeros((prop.shape[0], sc_data.shape[1]))
    allcellname = celltype_groups.keys()
    print(allcellname)
    print('Sampling cells to compose pseudo-bulk data')
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data[select_index].sum(axis=0)

    prop = pd.DataFrame(prop, columns=celltype_groups.keys())

    return sample, prop


# Split the data into target number
def trainTestSplit(X, Y, train_num, seed):
    np.random.seed(seed)

    X_num = X.shape[0]
    pre_index = list(range(X_num))
    src_index = []
    for i in range(train_num):
        randomIndex = int(np.random.uniform(0, len(pre_index)))  # Choose train set by random
        src_index.append(pre_index[randomIndex])
        del pre_index[randomIndex]
    src_data = X[src_index]
    src_label = Y[src_index]
    pre_data = X[pre_index]
    pre_label = Y[pre_index]
    return src_data, pre_data, src_label, pre_label

def h5adTodf(path):
    adata = sc.read_h5ad(path)
    sc_df = scv.DataFrame(adata.X) # index: cells, columns: genes.
    # sc_df = pd.DataFrame(adata.X.todense())

    genes = adata.var.index
    sc_df.columns = genes

    cells = adata.obs.index
    sc_df.index = cells

    sc_meta = adata.obs.loc[:,['cellID', 'cellType']]

    return sc_df, sc_meta



