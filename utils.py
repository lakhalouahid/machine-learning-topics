import random
import os
import time
import torch
import numpy as np

import matplotlib.pyplot as plt
from torch import nn, relu, softmax, optim
from torch.cuda import check_error
from torch.utils.data import TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def nmbr2vec(classes, n_classes=None):
    """
    Convert numpy array of number labels to numpy array of vector labels
    """
    classes_vec = np.zeros((classes.shape[0], n_classes))
    for i in range(classes.shape[0]):
        classes_vec[i, int(classes[i])] = 1
    return classes_vec


class FCNet(nn.Module):
    """
    Full Connected Neural network
    """
    def __init__(self, fc_layout=None, init=None):
        super().__init__()
        self.layers = []
        self.layers = nn.ModuleList(self.layers)
        for i in range(len(fc_layout) - 1):
            self.layers.append(nn.Linear(fc_layout[i], fc_layout[i+1]))
            if init == 'he':
                torch.nn.init.xavier_normal_(self.layers[i].weight)

    def forward(self, x):
        """
        Compute the outputs of layers
        """
        for i in range(len(self.layers) - 1):
            x = relu(self.layers[i](x))
        return softmax(self.layers[-1](x), dim=1)


class ConvNet(nn.Module):
    """
    Convolutional neural network
    """
    def __init__(self, cv_maxpool=None, cv_layout=None, cv_kernels=None, cv_strides=None, cv_padding=None, \
            cv_dropout=None, cv_batchnorm=None, use_avgpool=False, cv2linear_size=None, cv_lnlayout=None,
            cv_pdrop=None):
        """
        Initialise a convolutional neural network
        """
        super().__init__()
        self.cv_layers, self.fc_layers = [], []
        self.use_avgpool = use_avgpool
        self.cv_layers = nn.ModuleList(self.cv_layers)
        for i in range(len(cv_layout) - 1):
            self.cv_layers.append(nn.Sequential())
            self.cv_layers[-1].add_module("conv", nn.Conv2d(in_channels=cv_layout[i], out_channels=cv_layout[i+1], kernel_size=cv_kernels[i], stride=cv_strides[i], padding=cv_padding[i]))
            if cv_dropout[i]:
                self.cv_layers[-1].add_module("dropout", nn.Dropout2d(p=cv_pdrop[i], inplace=True))
            self.cv_layers[-1].add_module("relu", nn.ReLU(inplace=True))
            if cv_maxpool[i]:
                self.cv_layers[-1].add_module("maxpool", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
            if cv_batchnorm[i]:
                self.cv_layers[-1].add_module("batchnorm", nn.BatchNorm2d(cv_layout[i+1]))
        if use_avgpool:
            self.cv2linear = nn.AdaptiveAvgPool2d(cv2linear_size)
        self.fc_layers = nn.ModuleList(self.fc_layers)
        for i in range(len(cv_lnlayout)):
            if i == 0:
                self.fc_layers.append(nn.Linear(int(cv2linear_size[0]*cv2linear_size[1]*cv_layout[-1]), cv_lnlayout[i]))
            else:
                self.fc_layers.append(nn.Linear(cv_lnlayout[i-1], cv_lnlayout[i]))
        for l in self.modules():
            if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.Linear):
                nn.init.kaiming_normal_(l.weight.detach())
                l.bias.detach().zero_()

    def forward(self, x):
        """
        Compute the outputs of layers
        """
        for i in range(len(self.cv_layers)):
            x = self.cv_layers[i](x)
        x = torch.flatten(self.cv2linear(x), start_dim=1) if self.use_avgpool else torch.flatten(x, start_dim=1)
        for i in range(len(self.fc_layers)):
            x = self.fc_layers[i](x)
        return softmax(x, dim=1)



def fc(net, dataset, digit=2):
    """
    Round the accuracy of a network model for a dataset
    """
    return round(compute_accuracy(net, dataset)*100, digit)

def compute_accuracy(net, dataset):
    """
    compute the accuracy of a model for a dataset
    """
    with torch.no_grad():
        total, correct = 0, 0
        for batch in dataset:
            X, y = batch
            ypred = net(X)
            correct += torch.sum(torch.argmax(ypred, dim=1) == y)
            total += y.shape[0]
    return float(correct / total)

def seeds_rngs(seed=648712694):
    """
    Initialise all used rng for result repeatability
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic():
    """
    Set the behavior of torch to deterministic
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def resume_net(basename='', net=None, optimizer=None, scheduler=None, \
        prefix=None, checkpoints_path='model_checkpoints'):
    if basename == '':
        filepath = get_newest_file(checkpoints_path, prefix)
    else:
        filepath = os.path.join(checkpoints_path, basename)
    if filepath !=None and os.path.exists(filepath):
        print(f"Resuming from the checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        loss_history = checkpoint['loss_history']
        start_epoch = checkpoint['end_epoch']
        scheduler = checkpoint['scheduler']
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return {"net": net, "optimizer": optimizer, "scheduler": scheduler, "start_epoch": start_epoch, "loss_history": loss_history}
    else:
        print(f"Could not resume from from the checkpoint '{filepath}'")
    return None


def make_checkpoint(net, optimizer, scheduler, end_epoch, loss_history):
    checkpoint = {}
    checkpoint['end_epoch'] = end_epoch
    checkpoint['loss_history'] = loss_history
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['net'] = net.state_dict()
    checkpoint['scheduler'] = scheduler
    return checkpoint



def get_newest_file(path, prefix):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files if basename.find(prefix) == 0]
    if len(files)  == 0:
        return None
    return max(paths, key=os.path.getctime)

def save_checkpoint(checkpoint=None, prefix=None, loss=None, save_path='model_checkpoints'):
    print("Checkpoint start ...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, prefix + '_' + str(float(loss)) + '_' + str(time.time()))
    torch.save(checkpoint, filename)
    print("Checkpoint done ...")

def train_net(net=None, trainset=None, validset=None, loss_func=None, \
        optimizer=None, start_epoch=0, n_epoch=10, prefix=None, save_path=None, \
        batch_log=32, learn_decay=0.94, debug_epoch=1, debug=True, resume=False):
    """
    Train neural network
    """
    def fq(u,d):
        return f'{u:{len(str(d))}d}/{d:{len(str(d))}d}'
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learn_decay)
    n_batchs = len(trainset)
    if resume:
        cp = resume_net(net=net, optimizer=optimizer, scheduler=scheduler, prefix=prefix, checkpoints_path=save_path)
        if cp == None:
            loss_history = torch.zeros(start_epoch + n_epoch, 1)
        else:
            net = cp['net']
            optimizer = cp['optimizer']
            scheduler = cp['scheduler']
            start_epoch = cp['start_epoch']
            loss_history = torch.zeros(start_epoch + n_epoch, 1)
            loss_history[:start_epoch] = cp['loss_history']
    else:
        loss_history = torch.zeros(start_epoch + n_epoch, 1)
    loss = float(2<<32)
    for epoch_idx in range(start_epoch, start_epoch + n_epoch):
        for batch_idx, batch in enumerate(trainset):
            X, y = batch
            for param in net.parameters():
                param.grad = None
            ypred = net(X)
            loss = loss_func(ypred, y)
            loss.backward()
            optimizer.step()
            if debug and ((batch_idx+1) & (batch_log-1)) == 0:
                print (f'Epoch: {fq(epoch_idx+1, start_epoch + n_epoch)} | Batch: {fq(batch_idx+1, n_batchs)} | Cost: {float(loss):.6f}')
        scheduler.step()
        print(f'Epoch {epoch_idx+1:3.0f}/{start_epoch+n_epoch:3.0f}, loss: {float(loss):.6f} ------------------------')
        loss_history[epoch_idx] = float(loss)
        if (epoch_idx+1) & (debug_epoch-1) == 0:
            print(f'>> Cost: {float(loss):.6f}, Training accuracy: {fc(net, trainset)}, Validation accuracy: {fc(net, validset)}\n')
    checkpoint = make_checkpoint(net=net, optimizer=optimizer, loss_history=loss_history,\
            scheduler=scheduler, end_epoch=start_epoch+n_epoch)
    save_checkpoint(checkpoint=checkpoint, prefix=prefix, loss=loss, save_path=save_path)
    return loss_history


def split2tvt(all_X, all_y, train_ratio=0.8, valid_ratio=0.2, shuffle=False):
    """
    Split dataset to train, valid and test
    """
    if shuffle:
        perm = np.random.permutation(all_X.shape[0])
        all_X = all_X[perm, :]
        all_y = all_y[perm]
    total_samples = all_X.shape[0]
    ranges = (int(total_samples * train_ratio * (1-valid_ratio)), int(total_samples * train_ratio))
    all_X = torch.from_numpy(all_X).to(device, non_blocking=True)
    all_y = torch.from_numpy(all_y).to(device, non_blocking=True)
    return TensorDataset(all_X[:ranges[0], :], all_y[:ranges[0]]), \
            TensorDataset(all_X[ranges[0]:ranges[1], :], all_y[ranges[0]:ranges[1]]), \
            TensorDataset(all_X[ranges[1]:, :], all_y[ranges[1]:])

def display_samples(X, shape=None):
    pic_c, pic_w, pic_h = X.shape[1], X.shape[2], X.shape[3]
    image = np.zeros((shape[0]*pic_w, shape[1]*pic_h, pic_c))
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i*pic_h:(i+1)*pic_h, j*pic_w:(j+1)*pic_w] = np.swapaxes(X[i*shape[0]+j].swapaxes(1, 2), 0, 2)/255
    f = plt.figure(1, figsize=(14, 14))
    ax = f.add_subplot(1, 1, 1)
    ax.imshow(image)
    plt.show(block=False)

