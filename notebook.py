#!/usr/bin/env python

# %%
import time
import matplotlib.pyplot as plt
import torch
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
from torch import nn, optim
from torchwrapper.preprocessing import load_cifar10, normalize, split2tvt
from torchwrapper.architecture import CVModule
from torchwrapper.visualisation import display_samples, visualize_dataset
from torchwrapper.training import train
from torchwrapper.analysis import format_accuracy
from torch.utils.data import DataLoader

logging.root.setLevel(logging.DEBUG)


# %%
print('General setup ...')
n_epochs = 4
show_history = True
lr = 0.0001
lr_decay = float(1-1e-3)
batch_size = 1024
checkpoints_path = 'checkpoints_path'


# %%
print('Load and prepare data ...')
all_X, all_y = load_cifar10()
print(f"the shape of all_X {all_X.shape}")
print(f"the shape of all_y {all_y.shape}")


# %%
print('Reshaping the images ...')
all_X = all_X.reshape(-1, 3, 32, 32)
display_samples(all_X[:4*6], shape=(4, 6))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
visualize_dataset(all_X, all_y, class_names)


# %%
print('Normalization of the images ...')
all_X = normalize(all_X, trainset_ratio=0.8, training_ratio=0.8)

# %%
print('Split to trainset, validationset and testset ...')
train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y, train_ratio=0.8, shuffle=True)
trainset = DataLoader(train_X_y, batch_size=batch_size, shuffle=False)
validset = DataLoader(valid_X_y, batch_size=2*batch_size, shuffle=False)
testset = DataLoader(test_X_y, batch_size=2*batch_size, shuffle=False)
del all_X, all_y



# %%


# %%
print('Construting datasets ...')
cv_kernels = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3))
cv_strides = ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1))
cv_padding = ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1))
cv_layout = (3, 16, 64, 128, 128, 256, 256, 256)
cv_maxpool = [False, True, False, True, False, False, True]
cv_dropout = [False, False, False, False, False, False, False]
cv_batchnorm = [True, True, True, True, True, True, True]
cv2linear_size = (4, 4)
fc_layout  = (128, 10)
model = CVModule(cv_layout, cv_maxpool, cv_kernels, cv_strides, \
        cv_padding, cv_dropout, cv_batchnorm, cv2linear_size, fc_layout)

model.to(device)

# %%
print('Defining optimizer and cost function ...')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.92, 0.999), eps=1e-8, weight_decay=0.002, amsgrad=False)


# %%
print('Starting training ...')
start = time.time()
loss_history = train(model, trainset, validset, criterion, optimizer, 'classification', checkpoints_path, \
        start_epoch=0, n_epochs=n_epochs, batch_log=8, learn_decay=float(1-4e-4), debug_epoch=4, resume=False)
end = time.time()
print('Finishing training ...')

# %%
if show_history:
    f, a = plt.subplots()
    a.plot(loss_history.detach().numpy())
    f.canvas.draw()
    f.canvas.flush_events()
    plt.show()


# %%
print(f'Training time: {end-start}')


# %%
train_loss = format_accuracy(model, trainset)
valid_loss = format_accuracy(model, validset)
test_loss = format_accuracy(model, testset)


# %%
print(f'\nComputing the performance of the model...')
print(f'\nTraining accuracy: {train_loss}, Validation accuracy: {valid_loss}, Test accuracy: {test_loss}\n')
