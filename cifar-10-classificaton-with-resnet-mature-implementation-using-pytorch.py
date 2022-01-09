{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import matplotlib.pyplot as plt\n",
    "import torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch import nn, optim\n",
    "from utils import *\n",
    "from lib import load_raw_cifar10_data\n",
    "from torch.utils.data import DataLoader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('General setup ...')\n",
    "end_epoch = 256 # 50 epoch, use convolutional neural network\n",
    "show_history = False\n",
    "# seeds_rngs() # reset all random number generatars\n",
    "lr = 0.0001 # learning rate\n",
    "lr_decay = float(1-1e-3) # exponential decay learning rate per epoch\n",
    "batch_size = 1024 # batch size\n",
    "# save_path = '/content/drive/MyDrive/DeepLearningProjects/classification/cifar-10'\n",
    "save_path = 'model_checkpoints'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Load and prepare data ...')\n",
    "# this is wrapper function to load the images, and there labels of the training, validation and test sets to \n",
    "# two giants arrays,`all_X` contains all images, and `all_y` contains the labels\n",
    "all_X, all_y = load_raw_cifar10_data() \n",
    "print(f\"the shape of all_X {all_X.shape}\") # 50000 rgb images of size 32x32 pixels per channel\n",
    "print(f\"the shape of all_y {all_y.shape}\") # 50000 label vectors"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "if the neural network of type convolution, the shape of `all_X` ndarray should be depthxwithxheight"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_X = all_X / 255"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dataset visualisation<br>\n",
    "display_samples(all_X[:4*6], shape=(4, 6))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "normalization over the training set<br>\n",
    "train_samples = int((0.8**2)*all_X.shape[0]) # number of samples in the trainset<br>\n",
    "ll_X_mean = np.mean(all_X[:train_samples], axis=(0, 2, 3)).reshape(1, 3, 1, 1) # compute the mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_X_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)\n",
    "# all_X_std = np.std(all_X[:train_samples], axis=(0, 2, 3)).reshape(1, 3, 1, 1) # compute the std deviation\n",
    "all_X_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_X = (all_X - all_X_mean) / all_X_std # normalization of the whole dataset(train, validation, test)\n",
    "# all_X = np.asarray(all_X, dtype=np.float16)\n",
    "# all_y = np.asarray(all_y, dtype=np.float16)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Split the dataset to train, validation and test datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_X_y, valid_X_y, test_X_y = split2tvt(all_X, all_y, train_ratio=0.8, shuffle=True) # check the function defi\n",
    "del all_X, all_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Construting datasets ...')\n",
    "# Constructing the datasets, taking into account the batch size to use \n",
    "trainset = DataLoader(train_X_y, batch_size=batch_size, shuffle=False)\n",
    "validset = DataLoader(valid_X_y, batch_size=2*batch_size, shuffle=False)\n",
    "testset = DataLoader(test_X_y, batch_size=2*batch_size, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)\n",
    "net.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)\n",
    "net.to(device)\n",
    "print(f\"the network architecture: \\n{net}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Defining optimizer and cost function ...')\n",
    "loss_func = nn.MultiLabelSoftMarginLoss()\n",
    "optimizer = optim.Adam(params=net.parameters(), lr=lr, betas=(0.92, 0.999), \\\n",
    "        eps=1e-8, weight_decay=0.02, amsgrad=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Starting training ...')\n",
    "start = time.time()\n",
    "loss_history = train_net(net, trainset, validset, loss_func=loss_func, optimizer=optimizer,\n",
    "        start_epoch=0, end_epoch=512, batch_log=64, learn_decay=lr_decay, save_path=save_path, \\\n",
    "        debug=False, debug_epoch=8, prefix='classification-cifar-10', resume=False)\n",
    "end = time.time()\n",
    "print('Finishing training ...')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for g in optimizer.param_groups:\n",
    "    print(g['weight_decay'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if show_history:\n",
    "    f, a = plt.subplots()\n",
    "    a.plot(loss_history.detach().numpy())\n",
    "    f.canvas.draw()\n",
    "    f.canvas.flush_events()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Training time: {end-start}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loss = fc(net, trainset)\n",
    "valid_loss = fc(net, validset)\n",
    "test_loss = fc(net, testset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'\\n\\nComputing the performance of the model...')\n",
    "print(f'\\nTraining accuracy: {train_loss},        Validation accuracy: {valid_loss},        Test accuracy: {test_loss}\\n')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Saving model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "filename = ''\n",
    "total_loss_str = str(train_loss) + '_' + str(valid_loss) + '_' + str(test_loss)\n",
    "filename = 'models/net_cv_' + total_loss_str + str(time.time())\n",
    "torch.save(net, filename)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
