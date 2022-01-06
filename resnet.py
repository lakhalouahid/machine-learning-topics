import time
import matplotlib.pyplot as plt
import torch


from torch import nn, optim
from utils import *
from lib import load_raw_cifar10_data
from torch.utils.data import DataLoader

print('General setup ...')
end_epoch = 256 # 50 epoch, use convolutional neural network
show_history = False
# seeds_rngs() # reset all random number generatars
lr = 0.0001 # learning rate
lr_decay = float(1-1e-3) # exponential decay learning rate per epoch
batch_size = 16 # batch size
# save_path = '/content/drive/MyDrive/DeepLearningProjects/classification/cifar-10'
save_path = 'model_checkpoints'

print('Load and prepare data ...')
all_X, all_y = load_raw_cifar10_data(20000, 0)
print(f"the shape of all_X {all_X.shape}") # 50000 rgb images of size 32x32 pixels per channel
print(f"the shape of all_y {all_y.shape}") # 50000 label vectors

# if the neural network of type convolution, the shape of `all_X` ndarray should be depthxwithxheight

# Dataset visualisation
# display_samples(all_X[:4*6], shape=(4, 6))

# normalization over the training set
# train_samples = int((0.8**2)*all_X.shape[0]) # number of samples in the trainset
#all_X_mean = np.mean(all_X[:train_samples], axis=(0, 2, 3)).reshape(1, 3, 1, 1) # compute the mean
all_X_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
# all_X_std = np.std(all_X[:train_samples], axis=(0, 2, 3)).reshape(1, 3, 1, 1) # compute the std deviation
all_X_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

all_X = (all_X - all_X_mean) / all_X_std # normalization of the whole dataset(train, validation, test)
# all_X = np.asarray(all_X, dtype=np.float16)
# all_y = np.asarray(all_y, dtype=np.float16)

# Split the dataset to train, validation and test datasets
train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y, train_ratio=0.8, shuffle=True) # check the function defi
del all_X, all_y

print('Construting datasets ...')
# Constructing the datasets, taking into account the batch size to use 
trainset = DataLoader(train_X_y, batch_size=batch_size, shuffle=False)
validset = DataLoader(valid_X_y, batch_size=2*batch_size, shuffle=False)
testset = DataLoader(test_X_y, batch_size=2*batch_size, shuffle=False)

net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
net.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
net.to(device)
print(f"the network architecture: \n{net}")

print('Defining optimizer and cost function ...')
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.92, 0.999), \
        eps=1e-8, weight_decay=0.02, amsgrad=False)

print('Starting training ...')
start = time.time()
loss_history = train_net(net, trainset, validset, loss_func=loss_func, optimizer=optimizer,
        start_epoch=0, end_epoch=64, batch_log=64, learn_decay=lr_decay, save_path=save_path, \
        debug=False, debug_epoch=8, prefix='classification-cifar-10', resume=False)
end = time.time()
print('Finishing training ...')

for g in optimizer.param_groups:
    print(g['weight_decay'])

if show_history:
    f, a = plt.subplots()
    a.plot(loss_history.detach().numpy())
    f.canvas.draw()
    f.canvas.flush_events()
    plt.show()


print(f'Training time: {end-start}')

train_loss = fc(net, trainset)
valid_loss = fc(net, validset)
test_loss = fc(net, testset)

print(f'\n\nComputing the performance of the model...')
print(f'\nTraining accuracy: {train_loss},        Validation accuracy: {valid_loss},        Test accuracy: {test_loss}\n')

# Saving model
filename = ''
total_loss_str = str(train_loss) + '_' + str(valid_loss) + '_' + str(test_loss)
filename = 'models/net_cv_' + total_loss_str + str(time.time())
torch.save(net, filename)
