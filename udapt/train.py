import torch
import random
import warnings
import os
import time
from tqdm import tqdm, trange


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from utils import MyDataset, trainTestSplit, preprocess, ProcessInputData

from evaluation import CCCscore, output_eval
from model import AutoEncoder, discriminator\

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

####################################################### Utils functions ########################################################################
####################################################### Utils functions ########################################################################
####################################################### Utils functions ########################################################################
# Covert tensor to variable which can back propagation
def make_variable(tensor, volatile=False):
  ''' function to make tensor variable '''
  if use_cuda:
    tensor = tensor.cuda()
  return Variable(tensor, volatile=volatile)

def reproducibility(seed=9):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def plot_curse(data, epochs, ylabel_ = "Train loss curse"):
    x = range(epochs)
    y = data

    plt.plot(x, y, '.-')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel_)
    plt.show()

def pre_train(num_epochs, lr, save_step, autoencoder, train_loader):
    ###------ (1) Define loss function ------###
    ### Define loss for predictor ###
    criterion_predict = nn.L1Loss()

    optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=lr) # , betas=(0.5, 0.9)

    pred_loss = []
    recon_loss = []

    if use_cuda:
        autoencoder.cuda()

    for epoch in tqdm(range(num_epochs)):
        for j, (data, pred) in enumerate(train_loader):
            autoencoder.train()

            data = make_variable(data)
            pred = make_variable(pred)

            optimizer_autoencoder.zero_grad()

            ### Forward only SOURCE DOMAIN images through Encoder & Classifier ### # .float()
            x_recon_src, feat_x, feat_x1, sigmatrix_x = autoencoder(data.float())

            ### Calculate class-classification loss for Encoder and Classifier ###
            # batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data)
            loss_predict = criterion_predict(feat_x1, pred.float()) + criterion_predict(x_recon_src,data)
            loss_predict.backward()

            pred_loss.append(criterion_predict(feat_x1, pred.float()).cpu().detach().numpy())
            recon_loss.append(criterion_predict(x_recon_src,data).cpu().detach().numpy())

            optimizer_autoencoder.step()

    plot_curse(pred_loss, len(pred_loss), "Train loss curse of predictor")
    plot_curse(recon_loss, len(recon_loss), "Train loss curse of reconstruction")

    return autoencoder

def refraction(x):
    x_norm = np.zeros(x.shape)
    x_sum = np.sum(x, axis=1)
    for i in range(x.shape[0]):
        x_norm[i] = x[i] / x_sum[i]
    return x_norm


def train_adda(num_epochs, lr, save_step, autoencoder, discriminator, src_data_loader, tgt_data_loader, alpha_predict=1,alpha_DA=1, ip_dim=1, k=4):
    ''' ADDA Training using symmetric mapping '''

    ###------ (1) Define loss function ------###
    ### Define loss for predictor ###
    criterion_predict = nn.L1Loss()
    criterion_recon = nn.L1Loss()

    ### Define loss for domain-classification ###
    criterion_DA = nn.CrossEntropyLoss()

    ###------ (1) Define optimizer for model ------###
    decoder_parameters = [{'params': [p for n, p in autoencoder.named_parameters() if 'decoder' in n]}]
    encoder_parameters = [{'params': [p for n, p in autoencoder.named_parameters() if 'encoder' in n]}]
    predicter_parameters = [{'params': [p for n, p in autoencoder.named_parameters() if 'predicter' in n]}]

    optimizer_decoder = torch.optim.Adam(decoder_parameters, lr=lr) # , betas = (0.5, 0.9)
    optimizer_encoder = torch.optim.Adam(encoder_parameters, lr=lr) # , betas = (0.5, 0.9)
    optimizer_predicter = torch.optim.Adam(predicter_parameters, lr=lr) # , betas = (0.5, 0.9)

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_encoder_dis = torch.optim.Adam(encoder_parameters, lr=lr) # , betas = (0.5, 0.9)

    if use_cuda:
        autoencoder.cuda()
        discriminator.cuda()

    predict_loss_list = []
    domain_loss_list = []
    fake_domain_loss = []
    recon_src_list = []
    recon_tgt_list = []
    source_ccc = []

    for epoch in tqdm(range(num_epochs), position=0):
        for step, ((src_x, src_y), (tgt_x, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            ##########################################################################
            #######  2.1 Train Source Encoder & Classifier with class labels  ########
            ##########################################################################

            src_x, tgt_x = make_variable(src_x), make_variable(tgt_x)
            src_y = src_y.type(torch.FloatTensor).cuda()

            ### Forward only Source DOMAIN images through Encoder & Classifier & Decoder###
            autoencoder.train()

            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_predicter.zero_grad()

            x_recon_src, _ , feat_src1 , _ = autoencoder(src_x.float())
            x_recon_tgt, feat_tgt, _, _ = autoencoder(tgt_x.float())


            ### Calculate class-classification loss for Encoder and Classifier ###
            predict_loss_src = criterion_predict(feat_src1, src_y.float())
            recon_loss_src = criterion_predict(x_recon_src, src_x)
            recon_loss_tgt = criterion_predict(x_recon_tgt, tgt_x)
            recon_loss = recon_loss_src + recon_loss_tgt

            total_loss = alpha_predict * predict_loss_src + alpha_DA * recon_loss
            total_loss.backward()

            optimizer_encoder.step()
            optimizer_predicter.step()
            optimizer_decoder.step()

            recon_src_list.append(recon_loss_src.cpu().detach().numpy())
            recon_tgt_list.append(recon_loss_tgt.cpu().detach().numpy())

            predict_loss_list.append(predict_loss_src.cpu().detach().numpy())


            # optimizer_encoder.step()
            # optimizer_predicter.step()
            # optimizer_decoder.step()
            ##########################################################################
            #############  2.2 Train Discriminator with domain labels  ###############
            ##########################################################################
            discriminator.train()
            optimizer_discriminator.zero_grad()

            ### Forward pass through Encoder ###
            _, feat_src, _, _ = autoencoder(src_x.float())
            _, feat_tgt, _, _ = autoencoder(tgt_x.float())

            ### Concatenate source domain and target domain features ###
            feat_concat = torch.cat((feat_src, feat_tgt), 0)  # [batch_size*2, shape, 1, 1], concatenate
            feat_concat = feat_concat.squeeze(-1).squeeze(-1)  # [batch_size*2, shape]

            ### Forward concatenated features through Discriminator ###
            pred_concat = discriminator(feat_concat.detach()) # [batch_size*2, 2]

            ### prepare source domain labels (1) and target domain labels (0) ###
            label_src = make_variable(torch.ones(feat_src.size(0)).long()) # [batch_size, 1]
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long()) # [batch_size, 1]
            label_concat = torch.cat((label_src, label_tgt), 0) # [batch_size*2, 1]

            ### Calculate domain-classification loss for Discriminator ###
            loss_discriminator = criterion_DA(pred_concat.squeeze(1), label_concat)

            ### Backward Propagation for Discriminator ###
            loss_discriminator.backward()
            optimizer_discriminator.step()

            ### Update running losses/accuracies ###
            domain_loss_list.append(loss_discriminator.cpu().detach().numpy())

            ##########################################################################
            ############  2.3 Train Source Encoder w/ FAKE domain label  #############
            ##########################################################################

            ### Forward only TARGET DOMAIN images through Encoder ###
            # autoencoder.train()
            optimizer_encoder_dis.zero_grad()

            x_recon_tgt, feat_tgt, feat_tgt1, _ = autoencoder(tgt_x.float())

            ### Forward only TARGET DOMAIN features through Discriminator ###
            pred_tgt = discriminator(feat_tgt.squeeze(-1).squeeze(-1))
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())  # prepare fake labels, round down

            ### Calculate FAKE domain-classification loss for Encoder ###
            # loss_predict = criterion_predict(feat_src1, src_y.float()) + criterion_predict(x_recon_src,src_x) + criterion_predict(x_recon_tgt, tgt_x)

            loss_DA = criterion_DA(pred_tgt.squeeze(1), label_tgt)

            fake_domain_loss.append(loss_DA.cpu().detach().numpy())

            ### For encoder and Classifier,
            ### optimize class-classification & fake domain-classification losses together ###
            loss_DA.backward()
            optimizer_encoder_dis.step()

            source_ccc.append(CCCscore(feat_src1.detach().cpu(), src_y.cpu().float()))

    plot_curse(recon_src_list, len(recon_src_list), "Train loss curse of source reconstruction")
    plot_curse(recon_tgt_list, len(recon_tgt_list), "Train loss curse of target reconstruction")
    plot_curse(predict_loss_list, len(predict_loss_list), "Train loss curse of predictor")
    plot_curse(domain_loss_list, len(domain_loss_list), "Loss curse of discriminator")
    plot_curse(fake_domain_loss, len(fake_domain_loss), "Loss curse of fake loss")
    plot_curse(source_ccc, len(source_ccc), "Loss curse of source ccc score")

    return autoencoder, discriminator

def train_adda_v22(num_epochs, lr, save_step, autoencoder, discriminator, src_data_loader, tgt_data_loader, alpha_predict=1,alpha_DA=1, ip_dim=1, k=4):
    ''' ADDA Training using symmetric mapping '''

    ###------ (1) Define loss function ------###
    ### Define loss for predictor ###
    criterion_predict = nn.L1Loss()
    criterion_recon = nn.L1Loss()

    ### Define loss for domain-classification ###
    criterion_DA = nn.CrossEntropyLoss()

    ###------ (1) Define optimizer for model ------###
    decoder_parameters = [{'params': [p for n, p in autoencoder.named_parameters() if 'decoder' in n]}]
    encoder_parameters = [{'params': [p for n, p in autoencoder.named_parameters() if 'encoder' in n]}]
    predicter_parameters = [{'params': [p for n, p in autoencoder.named_parameters() if 'predicter' in n]}]

    optimizer_decoder = torch.optim.Adam(decoder_parameters, lr=lr) # , betas = (0.5, 0.9) , weight_decay=1e-3, betas = (0, 0.99)
    optimizer_encoder = torch.optim.Adam(encoder_parameters, lr=lr)
    optimizer_predicter = torch.optim.Adam(predicter_parameters, lr=lr)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_encoder_dis = torch.optim.Adam(encoder_parameters, lr=lr)

    if use_cuda:
        autoencoder.cuda()
        discriminator.cuda()

    predict_loss_list = []
    domain_loss_list = []
    fake_domain_loss = []
    recon_src_list = []
    recon_tgt_list = []
    source_ccc = []
    target_ccc = []
    alpha_predict_src = alpha_predict
    alpha_predict_tgt = alpha_predict


    for epoch in tqdm(range(num_epochs), position=0):
        for step, ((src_x, src_y), (tgt_x, tgt_y)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            ##########################################################################
            #######  2.1 Train Source Encoder & Classifier with class labels  ########
            ##########################################################################

            src_x, tgt_x = make_variable(src_x), make_variable(tgt_x)
            src_y = src_y.type(torch.FloatTensor).cuda()

            ### Forward only Source DOMAIN images through Encoder & Classifier & Decoder###
            autoencoder.train()
            optimizer_encoder.zero_grad()
            optimizer_predicter.zero_grad()
            optimizer_decoder.zero_grad()

            x_recon_src, _ , feat_src1 , _ = autoencoder(src_x.float())

            ### Calculate class-classification loss for Encoder and Classifier ###
            predict_loss_src = criterion_predict(feat_src1, src_y.float())
            recon_loss_src = criterion_predict(x_recon_src, src_x)

            total_loss_src = alpha_predict * predict_loss_src + alpha_DA * recon_loss_src
            total_loss_src.backward()

            optimizer_encoder.step()
            optimizer_predicter.step()
            optimizer_decoder.step()

            recon_src_list.append(recon_loss_src.cpu().detach().numpy())
            predict_loss_list.append(predict_loss_src.cpu().detach().numpy())


            ### Forward only Target DOMAIN images through Encoder & Classifier & Decoder###
            # autoencoder.train()
            # optimizer_autoencoder.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_predicter.zero_grad()
            optimizer_decoder.zero_grad()

            x_recon_tgt, feat_tgt, _, _ = autoencoder(tgt_x.float())

            ### Calculate class-classification loss for Encoder and Classifier ###
            recon_loss_tgt = criterion_predict(x_recon_tgt, tgt_x)

            total_loss_tgt = alpha_DA * recon_loss_tgt
            total_loss_tgt.backward()
            recon_tgt_list.append(total_loss_tgt.cpu().detach().numpy())

            # optimizer_autoencoder.step()
            optimizer_encoder.step()
            optimizer_predicter.step()
            optimizer_decoder.step()
            ##########################################################################
            #############  2.2 Train Discriminator with domain labels  ###############
            ##########################################################################
            discriminator.train()
            optimizer_discriminator.zero_grad()

            ### Forward pass through Encoder ###
            _, feat_src, _, _ = autoencoder(src_x.float())
            _, feat_tgt, _, _ = autoencoder(tgt_x.float())

            ### Concatenate source domain and target domain features ###
            feat_concat = torch.cat((feat_src, feat_tgt), 0)  # [batch_size*2, shape, 1, 1], concatenate
            feat_concat = feat_concat.squeeze(-1).squeeze(-1)  # [batch_size*2, shape]

            ### Forward concatenated features through Discriminator ###
            pred_concat = discriminator(feat_concat.detach()) # [batch_size*2, 2]

            ### prepare source domain labels (1) and target domain labels (0) ###
            label_src = make_variable(torch.ones(feat_src.size(0)).long()) # [batch_size, 1]
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long()) # [batch_size, 1]
            label_concat = torch.cat((label_src, label_tgt), 0) # [batch_size*2, 1]

            ### Calculate domain-classification loss for Discriminator ###
            loss_discriminator = criterion_DA(pred_concat.squeeze(1), label_concat)

            ### Backward Propagation for Discriminator ###
            loss_discriminator.backward()
            optimizer_discriminator.step()

            ### Update running losses/accuracies ###
            domain_loss_list.append(loss_discriminator.cpu().detach().numpy())

            ##########################################################################
            ############  2.3 Train Source Encoder w/ FAKE domain label  #############
            ##########################################################################

            ### Forward only TARGET DOMAIN images through Encoder ###
            # autoencoder.train()
            optimizer_encoder_dis.zero_grad()

            x_recon_tgt, feat_tgt, feat_tgt1, _ = autoencoder(tgt_x.float())

            ### Forward only TARGET DOMAIN features through Discriminator ###
            pred_tgt = discriminator(feat_tgt.squeeze(-1).squeeze(-1))
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())  # prepare fake labels, round down

            ### Calculate FAKE domain-classification loss for Encoder ###
            # loss_predict = criterion_predict(feat_src1, src_y.float()) + criterion_predict(x_recon_src,src_x) + criterion_predict(x_recon_tgt, tgt_x)

            loss_DA = criterion_DA(pred_tgt.squeeze(1), label_tgt)

            fake_domain_loss.append(loss_DA.cpu().detach().numpy())

            ### For encoder and Classifier,
            ### optimize class-classification & fake domain-classification losses together ###
            loss_DA.backward()
            optimizer_encoder_dis.step()

            source_ccc.append(CCCscore(feat_src1.detach().cpu(), src_y.cpu().float()))

            target_ccc.append(CCCscore(feat_tgt1.detach().cpu(), tgt_y.float()))

    plot_curse(recon_src_list, len(recon_src_list), "Train loss curse of source reconstruction")
    plot_curse(recon_tgt_list, len(recon_tgt_list), "Train loss curse of target reconstruction")
    plot_curse(predict_loss_list, len(predict_loss_list), "Train loss curse of predictor")
    plot_curse(domain_loss_list, len(domain_loss_list), "Loss curse of discriminator")
    plot_curse(fake_domain_loss, len(fake_domain_loss), "Loss curse of fake loss")
    plot_curse(source_ccc, len(source_ccc), "Loss curse of source ccc score")
    plot_curse(target_ccc, len(target_ccc), "Loss curse of target ccc score")

    return autoencoder, discriminator


def train_model(train, train_y, test, test_y, n, k, shape = 128, batch_size=128, seed=0, run=1, iters_pre =200, iters_fine=200, lr_pre = 1e-4, lr_train = 1e-4):
    reproducibility(seed)

    if isinstance(train_y, pd.DataFrame):
        train_y = np.asarray(train_y)

    if isinstance(test_y, pd.DataFrame):
        test_y = np.asarray(test_y)

    source_dt, pre_dt, source_prop, pre_prop = trainTestSplit(train, train_y, test.shape[0], seed=0)
    batch_size = 128

    epoch_pre = int(iters_pre / (len(pre_dt) / batch_size))
    if len(source_dt) < batch_size:
        epoch_fine = iters_fine
    else:
        epoch_fine = int(iters_fine / (len(source_dt) / batch_size))
    print(epoch_fine)


    pre_data = MyDataset(pre_dt, pre_prop)
    pre_trainloader = DataLoader(pre_data, batch_size=batch_size, shuffle=True)

    for inputs, labels in pre_trainloader:
        inputs, labels = inputs.to(device), labels.to(device)


    # construct model
    ### Define optimizers for encoder, predictor, and discriminator ###
    autoencoder_me = AutoEncoder(n, shape=shape, k=k)
    discriminator_me = discriminator(shape=shape)

    autoencoder_me.to(device)
    discriminator_me.to(device)

    autoencoder_me = pre_train(num_epochs=epoch_pre, lr=lr_pre, save_step=5, autoencoder=autoencoder_me,train_loader=pre_trainloader)

    ############------------ Fine tune model ------------############
    # load source and target data

    src_data = MyDataset(source_dt, source_prop)
    tgt_train = MyDataset(test, test_y)

    src_loader = DataLoader(src_data, batch_size=batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_train, batch_size=batch_size, shuffle=True) # False before

    for inputs, labels in src_loader:
        inputs, labels = inputs.to(device), labels.to(device)

    for inputs, labels in tgt_loader:
        inputs, labels = inputs.to(device), labels.to(device)

    # fine-tune model
    autoencoder_me.state = "train"
    # lr = 1e-4
    alpha_pre = 1
    alpha_da = 1

    autoencoder_me, discriminator_me = train_adda(num_epochs=epoch_fine, lr=lr_train, save_step=5,
                                                  autoencoder=autoencoder_me, discriminator=discriminator_me,
                                                  src_data_loader=src_loader,
                                                  tgt_data_loader=tgt_loader,
                                                  alpha_predict=alpha_pre,
                                                  alpha_DA=alpha_da,
                                                  ip_dim=n,
                                                  k=k)

    # autoencoder_me, discriminator_me = train_adda_v22(num_epochs=epoch_fine, lr=lr_train, save_step=5,
    #                                               autoencoder=autoencoder_me, discriminator=discriminator_me,
    #                                               src_data_loader=src_loader,
    #                                               tgt_data_loader=tgt_loader,
    #                                               alpha_predict=alpha_pre,
    #                                               alpha_DA=alpha_da,
    #                                               ip_dim=n,
    #                                               k=k)

    return autoencoder_me, discriminator_me


def predict(train_X, train_y, test_X, test_y, inter_genes, n, k, shape = 128, batch_size=128, seed=0, run=1, iters_pre =200, iters_fine=200, lr_pre = 1e-4, lr_train = 1e-4):
    # train model
    autoencoder_me, discriminator_me = train_model(train_X, train_y, test_X, test_y, n,k, shape, batch_size=128, seed=0, run=1, iters_pre =2000, iters_fine=100, lr_pre = 1e-4, lr_train = 1e-4)

    # CCC of target data
    ### predict target proportions
    autoencoder_me.eval()
    discriminator_me.eval()
    autoencoder_me.state = "test"

    data = torch.tensor(test_X).to(device).to(torch.float32)

    with torch.no_grad():
        x_recon, z1, z, sigmatrix = autoencoder_me(data)
        target_result = z.cpu().detach().numpy()
        target_sigmatrix = sigmatrix.cpu().detach().numpy()

    target_result_df = pd.DataFrame(target_result)
    target_result_df.columns = test_y.columns

    target_sigmatrix_df = pd.DataFrame(target_sigmatrix)
    print(target_sigmatrix_df.shape)
    target_sigmatrix_df.columns = inter_genes
    target_sigmatrix_df.index = test_y.columns

    return target_result_df, target_sigmatrix_df