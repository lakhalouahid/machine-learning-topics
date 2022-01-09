import cupy as np
from torchwrapper.preprocessing import load_cifar10


class FullConnecteNN():
    def __init__(self, lr, reg, mu, size_batch, layers):
        self.lr = lr
        self.reg = reg
        self.mu = mu
        self.size_batch = size_batch
        self.w = []
        self.b = []
        self.wv = []
        self.bv = []
        self.layers = len(layers) - 1
        for i in range(len(layers)-1):
            self.w.append(np.random.randn(layers[i], layers[i+1], dtype=np.float32) / np.sqrt(layers[i]/2))
            self.wv.append(np.zeros((layers[i], layers[i+1]), dtype=np.float32))
            self.b.append(np.zeros((1, layers[i+1]), dtype=np.float32))
            self.bv.append(np.zeros((1, layers[i+1]), dtype=np.float32))

    def eval(self, x):
        for i in range(self.layers):
            x = x @ self.w[i] + self.b[i]
            if i != (self.layers-1):
                x = x * (x>0)
        x = np.exp(x)
        x = x/np.sum(x, axis=1, keepdims=True, dtype=np.float32)
        return np.argmax(x, axis=1)

    def train(self, images, labels, n_epochs):
        n_batchs = int(np.ceil(images.shape[0]/self.size_batch))
        loss = np.zeros((n_epochs, n_batchs))
        for i in range(n_epochs):
            self.lr = max(1e-4, self.lr*0.99)
            loss[i, :] = self.train_batch(images, labels, n_batchs, self.lr)
            print(f"loss {loss[i][-1]:.6f}")
            if i & 7 == 0:
                print(f"train accuracy : {self.accuracy(images, labels)*100:.2f}")
    def accuracy(self, images, labels):
        correct, total = 0, images.shape[0]
        n_batchs = int(np.ceil(images.shape[0]/self.size_batch))
        for i in range(n_batchs):
            batch_range = range(i*self.size_batch, min((i+1)*self.size_batch, images.shape[0]))
            correct += np.sum(self.eval(images[batch_range]) == labels[batch_range])
        return correct/total

    def train_batch(self, images, labels, n_batchs, lr):
        loss = np.zeros((n_batchs))
        for i in range(n_batchs):
            e, z, h = [], [], None
            batch_range = range(i*self.size_batch, min((i+1)*self.size_batch, images.shape[0]))
            z.append(images[batch_range])
            y = labels[batch_range]
            for j in range(self.layers):
                h = z[-1] @ self.w[j]  + self.b[j]
                if j != (self.layers-1):
                    e.append(h>0)
                    z.append(h*e[-1])
            exp_scores = np.exp(h)
            probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True, dtype=np.float32)
            correct_logprobs = -np.log(probs[range(self.size_batch), y])
            reg_loss = 0
            for j in range(self.layers):
                reg_loss += 0.5*self.reg*np.mean(self.w[j]*self.w[j])
            loss[i] = np.mean(correct_logprobs, axis=0)
            dw, db, dt = [], [], []
            dt.append(probs)
            dt[-1][range(self.size_batch), y] -= 1
            dt[-1] /= self.size_batch
            for j in range(self.layers):
                dw.append(z[-(j+1)].T @ dt[j] + self.reg * self.w[-(j+1)] / (self.w[-(1+j)].shape[0]*self.w[-(1+j)].shape[1]))
                db.append(np.sum(dt[j], axis=0, keepdims=True, dtype=np.float32))
                if j < self.layers - 1:
                    dt.append(dt[-1] @ self.w[-(j+1)].T)

            for j in range(self.layers):
                self.wv[j] = self.mu*self.wv[j] - lr*dw[-(j+1)]
                self.w[j] += self.wv[j]
                self.bv[j] = self.mu*self.bv[j] - lr*db[-(j+1)]
                self.b[j] += self.bv[j]
        return loss


images, labels = load_cifar10()
images = np.array(images).reshape(-1, 3, 1024)
images_mean = np.mean(images, axis=(0, 2)).reshape(1, 3, 1)
images_std = np.mean(images, axis=(0, 2)).reshape(1, 3, 1)
images = (images - images_mean)/images_std
images = images.reshape(-1, 3072)
labels = np.array(labels)
four_layers_nn = FullConnecteNN(0.01, 0.001, 0.95, 1000, (3072, 256, 128, 10))
four_layers_nn.train(images, labels, 100)
