# coding: utf-8
import os, sys
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from os.path import abspath, join, pardir
import gc
import random
import time
import pickle

from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score

# import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

from torchvision import datasets
from torchvision import transforms

import warnings
warnings.simplefilter('ignore')


##########################
### SETTINGS
##########################

# Hyperparameters
num_epochs = 50
batch_size = 10
PATH = "/scratch/nabaruns/RADC"
LOGFILE = os.path.join(PATH, "logfile_sfcn.txt")

NUM_WORKERS = 1
learning_rate = 0.01
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")
# RANDOM_SEED = 1

outlist = []
class_list =[]

df = pd.read_csv("/home/nabaruns/119subcor1000.csv")
print(df.head())


print(df.describe())


# df['age_bin'] = pd.cut(df['age_at_visit'], [0, 70.01, 75.01, 80.01, 85.01, 90.01, 120.01], labels=['0', '1', '2', '3', '4', '5'])
# df['age_bin'] = pd.cut(df['age_at_visit'], [0, 80.01, 85.01, 120.01], labels=['0', '1', '2'])
# df['age_bin'] = pd.cut(df['age_at_visit'], [0, 75.01, 85.01, 120.01], labels=['0', '1', '2'])
# print(df.head())


def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop

# # SFCN+Rank ordinal


def label_to_levels(label, num_classes):
    levels = [1]*label + [0]*(num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels


def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)


def compute_mae_and_mse(model, data_loader, NUM_CLASSES, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features1, features2, targets) in enumerate(data_loader):

        ###################################
        ##### CORAL LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(levels_from_label)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###

        features1 = features1.to(device)
        features2 = features2.to(device)
        targets = targets.to(device)

        logits, probas = model(features1, features2)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
        del features1, features2, targets
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


class SFCN(nn.Module):
    def __init__(self, channel_number1=[10, 16, 32, 64, 64], channel_number2=[10, 16, 32, 64, 64, 64], output_dim=2, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number1)
        self.feature_extractor1 = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number1[i-1]
            out_channel = channel_number1[i]
            if i < n_layer-1:
                self.feature_extractor1.add_module('conv_%d' % i,
                                                  self.conv3_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor1.add_module('conv_%d' % i,
                                                  self.conv3_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))

        n_layer = len(channel_number2)
        self.feature_extractor2 = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number2[i-1]
            out_channel = channel_number2[i]
            if i < n_layer-1:
                self.feature_extractor2.add_module('conv_%d' % i,
                                                  self.conv2_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor2.add_module('conv_%d' % i,
                                                  self.conv2_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        
        self.avgpool1 = nn.AvgPool3d(2)
        self.fc1 = nn.Linear(512, 256, bias=True)
        self.avgpool2 = nn.AvgPool2d(4)
        self.fc2 = nn.Linear(576, 256, bias=True)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(output_dim-1).float())  


    @staticmethod
    def conv3_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    @staticmethod
    def conv2_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm2d(out_channel),
                nn.MaxPool2d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x1, x2):
        out = list()
        x1 = self.feature_extractor1(x1)
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        
        x2 = self.feature_extractor2(x2)
        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)
        
        x = torch.cat((x1, x2), 1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


# def plot_matrices(matrices, matrix_kind):
#     n_matrices = len(matrices)
#     fig = plt.figure(figsize=(n_matrices * 4, 4))
#     for n_subject, matrix in enumerate(matrices):
#         plt.subplot(1, n_matrices, n_subject + 1)
#         matrix = matrix.copy()  # avoid side effects
#         # Set diagonal to zero, for better visualization
#         np.fill_diagonal(matrix, 0)
#         vmax = np.max(np.abs(matrix))
#         title = '{0}, subject {1}'.format(matrix_kind, n_subject)
#         plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
#                              title=title, figure=fig, colorbar=False)

def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m/torch.max(m)
    return imp

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 


print("Reading T1w images...\n")
# data_dir = os.path.join(PATH, "anat")

# pooled_t1_img = []
# index_91 = []
# for i, f in enumerate(df['projid_fu_year']):
#     t1_img = nib.load(os.path.join(data_dir,f+".nii.gz"))
#     t1_data = np.array(t1_img.get_fdata())
#     if(len(t1_data)==91):
#         t1_data = t1_data/np.mean(t1_data)
#         t1_data = crop_center(t1_data, (70, 90, 70))
#         pooled_t1_img.append(t1_data)
#         index_91.append(i)
# print(np.shape(pooled_t1_img))

infile = open(os.path.join(PATH, "anat_all"),'rb')
pooled_t1_img = pickle.load(infile)
infile.close()
print(np.shape(pooled_t1_img))

infile = open(os.path.join(PATH, "index_91_all"),'rb')
index_91 = pickle.load(infile)
infile.close()
print(np.shape(index_91))


print("Reading fMRI correlation matrices...\n")
# data_dir = os.path.join(PATH, "Willard499")


# pooled_corr_mat = []
# for i, f in enumerate(df['projid_fu_year'][index_91]):
#     corr_mat = np.load(os.path.join(data_dir,f+".npy"))
#     pooled_corr_mat.append(corr_mat[0])
# print(np.shape(pooled_corr_mat))

infile = open(os.path.join(PATH, "Willard499_all"),'rb')
pooled_corr_mat = pickle.load(infile)
infile.close()
print(np.shape(pooled_corr_mat))


print(df['sex'][index_91].value_counts().to_frame())

sp = np.shape(pooled_t1_img)
pooled_t1_img = np.array(pooled_t1_img).reshape((sp[0], 1)+ sp[1:])

sp = np.shape(pooled_corr_mat)
pooled_corr_mat = np.array(pooled_corr_mat).reshape(sp[0], 1, sp[-2], sp[-1])

def train_model(rseed):

    RANDOM_STATE = rseed
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)

    X_index = [i for i in range(len(pooled_t1_img))]

    # For SFCN
    ylabels = df['sex'][index_91].values.astype(int)
    X, y = make_imbalance(
        np.array(X_index).reshape(-1, 1),
        ylabels,
        sampling_strategy={0: 96, 1: 96},
        random_state=RANDOM_STATE,
    )
    train_x, val_x, train_y, val_y = train_test_split(X.reshape(1,-1)[0], y, test_size = 0.3, random_state=RANDOM_STATE, stratify=y)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size = 0.4, random_state=RANDOM_STATE, stratify=val_y)
    print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape), (test_x.shape, test_y.shape))



    # Architecture
    NUM_CLASSES = len(np.unique(ylabels))

    # Other
    # importance_weights = torch.ones(NUM_CLASSES-1, dtype=torch.float)
    train_y_ages = torch.tensor(train_y, dtype=torch.float)

    IMP_WEIGHT = 0
    # Data-specific scheme
    if not IMP_WEIGHT:
        importance_weights = torch.ones(NUM_CLASSES-1, dtype=torch.float)
    elif IMP_WEIGHT == 1:
        importance_weights = task_importance_weights(train_y_ages)
        importance_weights = importance_weights[0:NUM_CLASSES-1]

    train_dataset = TensorDataset( Tensor(pooled_t1_img[train_x]), Tensor(pooled_corr_mat[train_x]), Tensor(train_y) )
    val_dataset = TensorDataset( Tensor(pooled_t1_img[val_x]), Tensor(pooled_corr_mat[val_x]), Tensor(val_y) )
    test_dataset = TensorDataset( Tensor(pooled_t1_img[test_x]), Tensor(pooled_corr_mat[test_x]), Tensor(test_y) )

    # del pooled_t1_img, pooled_corr_mat
    # gc.collect()

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            drop_last=False,
                            shuffle=False)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size, 
                            drop_last=False,
                            shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            drop_last=False,
                            shuffle=False)

    train_dataset_len = len(train_dataset)
    del train_dataset, val_dataset, test_dataset
    gc.collect()

    # # Checking the dataset
    # for images1, images2, labels in train_loader:  
    #     print('Image batch dimensions:', images1.shape, images2.shape)
    #     print('Image label dimensions:', labels.shape)
    #     break
    # for images1, images2, labels in test_loader:  
    #     print('Image batch dimensions:', images1.shape, images2.shape)
    #     print('Image label dimensions:', labels.shape)
    #     break

    importance_weights = importance_weights.to(DEVICE)
    BATCH_SIZE = batch_size

    model = SFCN()
    model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True) 

    # print(model)


    print("# parameters:", count_parameters(model))


    start_time = time.time()
    best_mae, best_rmse, best_epoch = 999, 999, -1
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features1, features2, targets) in enumerate(train_loader):

            ###################################
            ##### CORAL LABEL CONVERSION #####
            ###------------------------------START-----------------------------------###
            levels = []
            for label in targets:
                levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
                levels.append(levels_from_label)
            levels = torch.stack(levels)
            ###------------------------------END-------------------------------------###
            
            features1 = features1.to(DEVICE)
            features2 = features2.to(DEVICE)
            targets = targets.to(DEVICE)
            levels = levels.to(DEVICE)

            # print(features.size())
            # FORWARD AND BACK PROP
            logits, probas = model(features1, features2)
            # print(logits.size(), levels.size(), importance_weights.size())
            cost = cost_fn(logits, levels, importance_weights)
            
            # L1 regularization
            l1_lambda = 1e-7
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            cost = cost + l1_lambda * l1_norm
            
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 1:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                    % (epoch+1, num_epochs, batch_idx,
                        train_dataset_len//BATCH_SIZE, cost))
                # print(s)
                # with open(LOGFILE, 'a') as f:
                #     f.write('%s\n' % s)

        model.eval()
        with torch.set_grad_enabled(False):
            valid_mae, valid_mse = compute_mae_and_mse(model, val_loader, NUM_CLASSES, device=DEVICE)

        if valid_mae < best_mae:
            best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
            ########## SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(PATH, "best_model100_gen_"+str(RANDOM_STATE)+".pt"))


        s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
            valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
        print(s)
        # with open(LOGFILE, 'a') as f:
        #     f.write('%s\n' % s)
        del features1, features2, targets, levels


    ########## SAVE PREDICTIONS ######
    all_pred = []
    all_probas = []
    model.load_state_dict(torch.load(os.path.join(PATH, "best_model100_gen_"+str(RANDOM_STATE)+".pt")))
    with torch.set_grad_enabled(False):
        for batch_idx, (features1, features2, targets) in enumerate(test_loader):
            
            features1 = features1.to(DEVICE)
            features2 = features2.to(DEVICE)
            logits, probas = model(features1, features2)
            all_probas.append(probas)
            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)
            lst = [str(int(i)) for i in predicted_labels]
            all_pred.extend(lst)
            del features1, features2

    # model = model.cpu()
    # del model
    torch.cuda.empty_cache()

    all_pred = [int(x) for x in all_pred]

    ############## classification_report ###########
    print(classification_report(test_y, all_pred))
    precision,recall,fscore,support = score(test_y, all_pred,average='macro')

    outlist.append([RANDOM_STATE, best_mae.item(), best_rmse.item(), best_epoch, precision,recall,fscore])

    ############## plot save #######################
    a = np.zeros(shape=(NUM_CLASSES,NUM_CLASSES))
    for i,pred in enumerate(all_pred):
        a[test_y[i]][pred] += 1
    print("actual cols vs pred rows:\n",a)
    class_list.append(a)

    # plt.scatter(all_pred, np.array(df['age_at_visit'][index_91])[test_x])
    # plt.xlabel("prediction")
    # plt.ylabel("sex")
    # plt.savefig("/home/nabarun/RADC_seed"+str(RANDOM_STATE)+"_actualVsPred_gen.png")
    # plt.clf()


# for i in range(100):
#     print("iter ", i)
    # rs = random.randint(3, 1000)
    
train_model(int(sys.argv[1]))

np.save("outlist_gender", outlist)
np.save("class_list_gender", class_list)