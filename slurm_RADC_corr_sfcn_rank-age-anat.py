import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, join, pardir
import gc
import random
import pickle

from imblearn.datasets import make_imbalance
from scipy.stats import norm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

import nibabel as nib


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

RANDOM_STATE = int(sys.argv[1])
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

inPATH = "/scratch/nabaruns/RADC"
homePATH = "/home/nabaruns"

df = pd.read_csv(os.path.join(homePATH, "radc_age_gender_labels.csv"))

# df['age_bin'] = pd.cut(df['age_at_visit'], [0, 70.01, 75.01, 80.01, 85.01, 90.01, 120.01], labels=['0', '1', '2', '3', '4', '5'])
df['age_bin'] = pd.cut(df['age_at_visit'], [0, 75.01, 80.01, 85.01, 120.01], labels=['0', '1', '2', '3'])
print(df.head())


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


# In[8]:


# data_dir = "/4tb/nabarun/RADC/fmriprep/anat"

# pooled_corr_mat=[]
# index_91 = []
# for i, f in enumerate(df['projid_fu_year']):
# #     corr_mat = np.load(os.path.join(data_dir,f+".npy"))
# #     pooled_corr_mat.append(corr_mat[0])
#     t1_img = nib.load(os.path.join(data_dir,f+".nii.gz"))
#     t1_data = np.array(t1_img.get_fdata())
#     if(len(t1_data)==91):
#         t1_data = t1_data/np.mean(t1_data)
#         t1_data = crop_center(t1_data, (70, 90, 70))
#         pooled_corr_mat.append(t1_data)
#         index_91.append(i)
# # pooled_corr_mat = np.array(np.load("pooled_corr_mat400.npy"))
# print(np.shape(pooled_corr_mat))


infile = open(os.path.join(inPATH, "anat_all_nan"),'rb')
pooled_t1_img = pickle.load(infile)
infile.close()
print(np.shape(pooled_t1_img))

infile = open(os.path.join(inPATH, "index_91pnan"),'rb')
index_91 = pickle.load(infile)
infile.close()
print(np.shape(index_91))

infile = open(os.path.join(inPATH, "Willard499_all_nan"),'rb')
pooled_corr_mat = pickle.load(infile)
infile.close()
print(np.shape(pooled_corr_mat))



def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)



# l = len(pooled_corr_mat)
# sp = np.shape(pooled_corr_mat[0])
# pooled_corr_mat = np.array(pooled_corr_mat).reshape((l, 1)+ sp)

sp = np.shape(pooled_t1_img)
pooled_t1_img = np.array(pooled_t1_img).reshape((sp[0], 1)+ sp[1:])

# # For coral-cnn
# ages = df['age_bin'].values.astype(int)

# X, y = make_imbalance(
#     pooled_corr_mat,
#     ages,
#     sampling_strategy={0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50},
#     random_state=RANDOM_STATE,
# )

# train_x, val_x, train_y, val_y = train_test_split(X, y, test_size = 0.3, stratify=ages)
# val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size = 0.4, stratify=val_y)
# (train_x.shape, train_y.shape), (val_x.shape, val_y.shape), (test_x.shape, test_y.shape)

X_index = [i for i in range(len(pooled_t1_img))]

# For SFCN
ages = df['age_bin'][index_91].values.astype(int)
X, y = make_imbalance(
    np.array(X_index).reshape(-1, 1),
    ages,
    sampling_strategy={0: 118, 1: 150, 2: 150, 3: 118},
    random_state=RANDOM_STATE,
)
train_x, val_x, train_y, val_y = train_test_split(pooled_t1_img[X.reshape(1,-1)[0]], y, random_state=RANDOM_STATE, test_size = 0.2, stratify=y)
val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size = 0.5, random_state=RANDOM_STATE, stratify=val_y)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape), (test_x.shape, test_y.shape)


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


##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.01
num_epochs = 50
batch_size = 10
PATH = "./"

# Architecture
NUM_CLASSES = len(np.unique(ages))

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




# Note transforms.ToTensor() scales input images
# to 0-1 range
# train_dataset = datasets.MNIST(root='mnist_data', 
#                                train=True, 
#                                transform=transforms.ToTensor(),
#                                download=True)

# test_dataset = datasets.MNIST(root='mnist_data', 
#                               train=False, 
#                               transform=transforms.ToTensor())

train_dataset = TensorDataset( Tensor(train_x), Tensor(train_y) )
val_dataset = TensorDataset( Tensor(val_x), Tensor(val_y) )
test_dataset = TensorDataset( Tensor(test_x), Tensor(test_y) )

del train_x, train_y, val_x, val_y, test_x

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          drop_last=False,
                          shuffle=True)

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

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
for images, labels in test_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# # SFCN+Rank ordinal

# In[20]:


def label_to_levels(label, num_classes):
    levels = [1]*label + [0]*(num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels


def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)


# In[23]:


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets) in enumerate(data_loader):

        ###################################
        ##### CORAL LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(levels_from_label)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 128, 64], output_dim=4, dropout=True):
#     def __init__(self, channel_number=[10, 16, 32, 64, 64], output_dim=4, dropout=True):
        
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
#         self.classifier = nn.Sequential()
#         avg_shape = [4, 4]
#         self.classifier.add_module('average_pool', nn.AvgPool2d(avg_shape))
#         if dropout is True:
#             self.classifier.add_module('dropout', nn.Dropout(0.5))
#         i = n_layer
#         in_channel = channel_number[-1]
#         out_channel = output_dim
#         self.classifier.add_module('conv_%d' % i,
#                                    nn.Conv2d(in_channel, out_channel, padding=0, kernel_size=1))
        
        self.avgpool = nn.AvgPool3d(2)
        self.fc = nn.Linear(512, 1, bias=True)
        self.linear_1_bias = nn.Parameter(torch.zeros(output_dim-1).float())

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
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

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
#         print(x_f.shape)
#         x = self.classifier(x_f)
#         x = F.log_softmax(x, dim=1)
#         out.append(x)
#         return out
        x = self.avgpool(x_f)
#         print(x.shape)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


# In[25]:


NUM_WORKERS = 1
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

importance_weights = importance_weights.to(DEVICE)
BATCH_SIZE = batch_size

model = SFCN()
model = torch.nn.DataParallel(model)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

print(model)


# In[26]:


start_time = time.time()
best_mae, best_rmse, best_epoch = 999, 999, -1
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        ###################################
        ##### CORAL LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(levels_from_label)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        levels = levels.to(DEVICE)

#         print(features.size())
        # FORWARD AND BACK PROP
        logits, probas = model(features)
        # print(logits.size(), levels.size(), importance_weights.size())
        cost = cost_fn(logits, levels, importance_weights)
        optimizer.zero_grad()

        cost.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
#         if not batch_idx % 1:
#             s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
#                  % (epoch+1, num_epochs, batch_idx,
#                      train_dataset_len//BATCH_SIZE, cost))
#             print(s)
#             with open(LOGFILE, 'a') as f:
#                 f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):
        valid_mae, valid_mse = compute_mae_and_mse(model, val_loader,
                                                   device=DEVICE)

    if valid_mae < best_mae:
        best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
        ########## SAVE MODEL #############
        torch.save(model.state_dict(), os.path.join(inPATH, 'best_model_anat.pt'))


    s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d | Time elapsed: %.2f min' % (
        valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch, ((time.time() - start_time)/60))
    print(s)
#     with open(LOGFILE, 'a') as f:
#         f.write('%s\n' % s)

#     s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
#     print(s)
#     with open(LOGFILE, 'a') as f:
#         f.write('%s\n' % s)


# In[27]:


########## SAVE PREDICTIONS ######
all_pred = []
all_probas = []
model.load_state_dict(torch.load(os.path.join(inPATH, 'best_model_anat.pt')))
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets) in enumerate(test_loader):
        
        features = features.to(DEVICE)
        logits, probas = model(features)
        all_probas.append(probas)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(lst)


# In[28]:


all_pred = [int(x) for x in all_pred]

from sklearn.metrics import classification_report
print(classification_report(test_y, all_pred))


# In[29]:


print(np.array(all_pred), test_y)

