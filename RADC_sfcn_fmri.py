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

RANDOM_STATE = int(sys.argv[1])+71
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)

inPATH = "/scratch/nabaruns/RADC"
homePATH = "/home/nabaruns"

df = pd.read_csv(os.path.join(homePATH, "radc_age_gender_labels.csv"))

bins = [i for i in range(64,96)]
labels = [i for i in range(31)]
df['age_bin'] = pd.cut(df['age_at_visit'], bins, labels=labels)
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


# index_nan = np.array(df[df['age_bin'].isnull()].index)

# data_dir = "/4tb/nabarun/RADC/fmriprep/anat"

# pooled_t1_img = []
# index_91 = []
# for i, f in enumerate(df['projid_fu_year']):
# #     corr_mat = np.load(os.path.join(data_dir,f+".npy"))
# #     pooled_corr_mat.append(corr_mat[0])
#     t1_img = nib.load(os.path.join(data_dir,f+".nii.gz"))
#     t1_data = np.array(t1_img.get_fdata())
#     if((len(t1_data)==91) and (i not in index_nan)):
#         t1_data = t1_data/np.mean(t1_data)
#         t1_data = crop_center(t1_data, (70, 90, 70))
#         pooled_t1_img.append(t1_data)
#         index_91.append(i)
# # pooled_corr_mat = np.array(np.load("pooled_corr_mat400.npy"))
# print(np.shape(pooled_t1_img))


# df['projid_fu_year'][index_91].to_csv("index91nan")

# filename = 'anat_all_nan'
# outfile = open(filename,'wb')
# pickle.dump(pooled_t1_img,outfile)
# outfile.close()

# filename = 'index_91pnan'
# outfile = open(filename,'wb')
# pickle.dump(index_91,outfile)
# outfile.close()


infile = open(os.path.join(inPATH, "anat_all_nan"),'rb')
pooled_t1_img = pickle.load(infile)
infile.close()
print(np.shape(pooled_t1_img))

infile = open(os.path.join(inPATH, "index_91pnan"),'rb')
index_91 = pickle.load(infile)
infile.close()
print(np.shape(index_91))


# data_dir = "/4tb/nabarun/RADC/corr_mat/Willard499"

# pooled_corr_mat = []
# for i, f in enumerate(df['projid_fu_year'][index_91]):
#     corr_mat = np.load(os.path.join(data_dir,f+".npy"))
#     pooled_corr_mat.append(corr_mat[0])
# print(np.shape(pooled_corr_mat))


# import pickle
# filename = 'Willard499_all_nan'
# outfile = open(filename,'wb')
# pickle.dump(pooled_corr_mat,outfile)
# outfile.close()


infile = open(os.path.join(inPATH, "Willard499_all_nan"),'rb')
pooled_corr_mat = pickle.load(infile)
infile.close()
print(np.shape(pooled_corr_mat))


# In[19]:


# df.dropna(inplace=True)
# for idx in df[df["sex"].isnull()].index:
#     pooled_corr_mat = np.delete(pooled_corr_mat, idx)


# In[20]:


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


sp = np.shape(pooled_t1_img)
pooled_t1_img = np.array(pooled_t1_img).reshape((sp[0], 1)+ sp[1:])

sp = np.shape(pooled_corr_mat)
pooled_corr_mat = np.array(pooled_corr_mat).reshape(sp[0], 1, sp[-2], sp[-1])

X = np.array([i for i in range(len(pooled_t1_img))])

# For SFCN
ylabels = df['age_bin'][index_91].values.astype(int)
# X, y = make_imbalance(
#     np.array(X).reshape(-1, 1),
#     y,
#     sampling_strategy={0: 78, 1: 120, 2: 120},
#     random_state=RANDOM_STATE,
# )
train_x, test_x, train_y, test_y = train_test_split(X, ylabels, test_size = 0.2, random_state=RANDOM_STATE, stratify=ylabels)
# val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size = 0.6, random_state=RANDOM_STATE)
# print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape), (test_x.shape, test_y.shape))
print((train_x.shape, train_y.shape), (test_x.shape, test_y.shape))


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


# In[25]:


##########################
### SETTINGS
##########################

# Hyperparameters
NUM_CLASSES = len(np.unique(ylabels))
batch_size = 10
bin_range = [0,NUM_CLASSES]
bin_step = 1
sigma = 1

# Other
# importance_weights = torch.ones(NUM_CLASSES-1, dtype=torch.float)
train_y_ages = torch.tensor(train_y, dtype=torch.float)

IMP_WEIGHT = 1
# Data-specific scheme
if not IMP_WEIGHT:
    importance_weights = torch.ones(NUM_CLASSES-1, dtype=torch.float)
elif IMP_WEIGHT == 1:
    importance_weights = task_importance_weights(train_y_ages)
    importance_weights = importance_weights[0:NUM_CLASSES-1]

# train_dataset = TensorDataset( Tensor(pooled_t1_img[train_x]), Tensor(train_y) )
# val_dataset = TensorDataset( Tensor(pooled_t1_img[val_x]), Tensor(val_y) )
test_dataset = TensorDataset( Tensor(pooled_corr_mat[test_x]), Tensor(test_y) )

# del pooled_t1_img, pooled_corr_mat
# gc.collect()

# train_loader = DataLoader(dataset=train_dataset, 
#                           batch_size=batch_size, 
#                           drop_last=False,
#                           shuffle=False)

# val_loader = DataLoader(dataset=val_dataset, 
#                          batch_size=batch_size, 
#                          drop_last=False,
#                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         drop_last=False,
                         shuffle=False)

# train_dataset_len = len(train_dataset)
# del train_dataset, val_dataset, test_dataset
# gc.collect()

# # Checking the dataset
# for images1, labels in train_loader:  
#     print('Image batch dimensions:', images1.shape)
#     print('Image label dimensions:', labels.shape)
#     break
for images1, labels in test_loader:  
    print('Image batch dimensions:', images1.shape)
    print('Image label dimensions:', labels.shape)
    break


# # SFCN+Rank ordinal

# In[26]:


def label_to_levels(label, num_classes):
    levels = [1]*label + [0]*(num_classes - 1 - label)
    levels = torch.tensor(levels, dtype=torch.float32)
    return levels


def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)


# In[32]:


def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    #print(loss)
    return loss


# In[33]:


def compute_cost(model, data_loader, device):
    loss, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):

        ###################################
        ##### CORAL LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            y, bc = num2vect(int(label.item()), bin_range, bin_step, sigma)
            y = torch.tensor(y, dtype=torch.float32)
            # levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(y)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###
        levels = levels.to(device)
        features = features.to(device)
        targets = targets.to(device)

        output = model(features)
        x = output[0].reshape(levels.size())
        cost = my_KLDivLoss(x, levels)
        num_examples += targets.size(0)
        loss += my_KLDivLoss(x, levels)
    loss = loss.float() / num_examples
    return loss


# In[34]:


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets) in enumerate(data_loader):

        ###################################
        ##### CORAL LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            y, bc = num2vect(int(label.item()), bin_range, bin_step, sigma)
            bc = torch.tensor(bc, dtype=torch.float32)
            # levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(bc)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###

        levels = levels.to(device)
        features = features.to(device)
        targets = targets.to(device)

        predicted_labels = []
        output = model(features)
        x = output[0].cpu().reshape(levels.size())
        prob = np.exp(x)
        prob = prob.to(device)
        # pred = prob@levels
        for i,bc in enumerate(levels):
            pred = prob[i]@bc
            pred = torch.tensor(pred, dtype=torch.float32)
            predicted_labels.append(pred)
        num_examples += targets.size(0)
        predicted_labels = torch.stack(predicted_labels)
#         print(predicted_labels, targets)
        mae += torch.abs(predicted_labels - targets).sum().data
        mse += ((predicted_labels - targets)*(predicted_labels - targets)).sum().data
        del features, targets
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


# In[35]:


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# In[36]:


class SFCN2D(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=31, dropout=True):
        super(SFCN2D, self).__init__()
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
        self.classifier = nn.Sequential()
#         avg_shape = [6, 6]
        self.classifier.add_module('average_pool', nn.AvgPool2d(8))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv2d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
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

    def forward(self, x):
        out = list()
#         print(x.shape)
        x_f = self.feature_extractor(x)
#         print(x_f.shape)
        x = self.classifier(x_f)
#         print(x.shape)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out


# In[37]:


class SFCN3D(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 128, 64], output_dim=31, dropout=True):
        super(SFCN3D, self).__init__()
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
        self.classifier = nn.Sequential()
        avg_shape = [3, 3, 3]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

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
#         print(x.shape)
        x_f = self.feature_extractor(x)
#         print(x_f.shape)
        x = self.classifier(x_f)
#         print(x.shape)
        x = F.log_softmax(x, dim=1)
        out.append(x)
        return out


# In[38]:


NUM_WORKERS = 1
num_epochs = 50
learning_rate = 0.01
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")
# RANDOM_SEED = 1

importance_weights = importance_weights.to(DEVICE)
BATCH_SIZE = batch_size

model = SFCN2D()
model = torch.nn.DataParallel(model)
model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3)
print(model)


def train_model(model,optimizer,train_loader,epoch=1):
    for batch_idx, (features, targets) in enumerate(train_loader):

        ###################################
        ##### num2vect LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            # print(int(label.item()))
            y, bc = num2vect(int(label.item()), bin_range, bin_step, sigma)
            y = torch.tensor(y, dtype=torch.float32)
            # levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(y)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
        levels = levels.to(DEVICE)

        # FORWARD AND BACK PROP
        output = model(features)
#         print(output.size(), levels.size(), importance_weights.size())
        x = output[0].reshape(levels.size())
        # print(logits.size(), levels.size(), importance_weights.size())
        # print(x, levels)
        # break
        cost = my_KLDivLoss(x, levels)
        optimizer.zero_grad()
        cost.backward()
        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
#         if not batch_idx % 5:
#             s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
#                  % (epoch+1, num_epochs, batch_idx,
#                      train_dataset_len//batch_size, cost))
#             print(s)
            # with open(LOGFILE, 'a') as f:
            #     f.write('%s\n' % s)


# In[42]:


numfold = 10
foldidx = 1
kf = KFold(n_splits=numfold, shuffle=True, random_state=RANDOM_STATE)
start_time = time.time()
best_mae, best_rmse, best_epoch = 999, 999, -1

for fold, (train_index, val_index) in enumerate(kf.split(train_x)):
    print('--------------------------------') 
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    train_dataset = TensorDataset( Tensor(pooled_corr_mat[train_index]), Tensor(ylabels[train_index]) )
    val_dataset = TensorDataset( Tensor(pooled_corr_mat[val_index]), Tensor(ylabels[val_index]) )

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              drop_last=False,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, 
                             batch_size=batch_size, 
                             drop_last=False,
                             shuffle=False)

    train_dataset_len = len(train_dataset)
    
    for epoch in range(num_epochs):
        model.train()
        train_model(model,optimizer,train_loader,epoch)

        model.eval()
        with torch.set_grad_enabled(False):
            valid_mae, valid_mse = compute_mae_and_mse(model, val_loader,
                                                       device=DEVICE)

        if valid_mae < best_mae:
            best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
            torch.save(model.state_dict(), os.path.join(inPATH, f'best_model_sfcn_corr_seed_{RANDOM_STATE}.pt'))

        scheduler.step(valid_mae)
        s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d | Time elapsed: %.2f min' % (
            valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch, ((time.time() - start_time)/60))
        print(s)
        # with open(LOGFILE, 'a') as f:
        #     f.write('%s\n' % s)


# In[43]:


########## SAVE PREDICTIONS ######
all_pred = []
all_probas = []
model.load_state_dict(torch.load(os.path.join(inPATH, f'best_model_sfcn_corr_seed_{RANDOM_STATE}.pt')))
model.eval()
with torch.set_grad_enabled(False):
    test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                               device=DEVICE)
s = 'Test MAE/RMSE: %.2f/%.2f ' % (
        test_mae, torch.sqrt(test_mse))
print(s)
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets) in enumerate(test_loader):

        ###################################
        ##### num2vect LABEL CONVERSION #####
        ###------------------------------START-----------------------------------###
        levels = []
        for label in targets:
            # print(int(label.item()))
            y, bc = num2vect(int(label.item()), bin_range, bin_step, sigma)
            # print(int(label.item()),bc)
            bc = torch.tensor(bc, dtype=torch.float32)
            # levels_from_label = label_to_levels(int(label.item()), NUM_CLASSES)
            levels.append(bc)
        levels = torch.stack(levels)
        ###------------------------------END-------------------------------------###
        
        predicted_labels = []
        output = model(features)
        x = output[0].cpu().reshape(levels.size())
        prob = np.exp(x)
        for i,bc in enumerate(levels):
            pred = prob[i]@bc
            pred = torch.tensor(pred, dtype=torch.float32)
            predicted_labels.append(pred)
        lst = [int(i) for i in predicted_labels]
        all_pred.extend(lst)


# In[44]:

print("test_pred:", test_y, np.array(all_pred))


# In[45]:


# plt.scatter(test_y[:]+64, np.array(all_pred)[:]+64)
# plt.ylabel("prediction")
# plt.xlabel("age_at_visit")
# plt.show()
