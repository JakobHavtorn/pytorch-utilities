import torch
import torch.nn as nn
from torch.autograd import Variable


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()
        
        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)
        
    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = self.log_alpha.exp()
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3
        
        kl = -negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x


class Solver(object):
    def __init__(self, Network=CNN, dropout_method='standard', dataset='MNIST', lr=0.005):        
        self.train_loader, self.test_loader = build_dataset(dataset, './data')
        
        self.image_dim = {'MNIST': (28, 28), 'CIFAR10': (3, 32, 32)}[dataset]
        
        self.dropout_method = dropout_method
        
        self.net = Network(
            image_dim=self.image_dim,
            dropout_method=dropout_method).cuda()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def train(self, n_epochs=50):
        self.net.train()
        
        for epoch_i in range(n_epochs):
            epoch_i += 1
            epoch_loss = 0
            epoch_kl = 0
            for images, labels in self.train_loader:
                images = Variable(images).view(-1, self.image_dim).cuda()
                labels = Variable(labels).cuda()

                logits = self.net(images)
                
                
                loss = self.loss_fn(logits, labels)
                
                if self.dropout_method == 'variational':
                    kl = self.net.kl()
                    total_loss = loss + kl / 10
                else:
                    total_loss = loss

                self.optimizer.zero_grad()
                total_loss.backward()

                self.optimizer.step()
                
                epoch_loss += float(loss.data)
                if self.dropout_method == 'variational':
                    epoch_kl += float(kl.data)

            if not self.dropout_method == 'variational':
                epoch_loss /= len(self.train_loader.dataset)
                print('Epoch {epoch_i} | loss: {epoch_loss:.4f}'.format(epoch_i=epoch_i, epoch_loss=epoch_loss))
            else:
                epoch_loss /= len(self.train_loader.dataset)
                epoch_kl /= len(self.train_loader.dataset)
                print('Epoch {epoch_i} | loss: {epoch_loss:.4f}, kl: {epoch_kl:.4f}'.format(epoch_i=epoch_i, epoch_loss=epoch_loss, epoch_kl=epoch_kl))
            
    def evaluate(self):
        total = 0
        correct = 0
        self.net.eval()
        for images, labels in self.test_loader:
            images = Variable(images).view(-1, self.image_dim).cuda()

            logits = self.net(images)
            
            _, predicted = torch.max(logits.data, 1)
            
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
                
        print('Accuracy: {acc:.2f}%'.format(acc=100 * correct / total))


class CNN(nn.Module):
    def __init__(self, image_dim=(28, 28), dropout_method='standard'):
        super(CNN, self).__init__()
        """3-Layer Fully-connected NN"""
        def dropout(p=None, dim=None, method='standard'):
        if method == 'standard':
            return nn.Dropout(p)
        elif method == 'gaussian':
            return GaussianDropout(p/(1-p))
        elif method == 'variational':
            return VariationalDropout(p/(1-p), dim)
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=None, padding=0, dilation=1),
            nn.ReLU(),
            nn.Linear(320, 256),
            dropout(0.2, 256, dropout_method),
            nn.ReLU(),
            nn.Linear(256, 100),
            dropout(0.5, 100, dropout_method),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.SoftMax()
        )
        
    def kl(self):
        kl = 0
        for name, module in self.net.named_modules():
            if isinstance(module, VariationalDropout):
                kl += module.kl().sum()
        return kl
        
            
    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    print("Standard dropout")
    standard_solver = Solver(dropout_method='standard')
    standard_solver.train(n_epochs=50)
    standard_solver.evaluate()

    print("Gaussian dropout")
    gaussian_solver = Solver(dropout_method='gaussian')
    gaussian_solver.train(n_epochs=50)
    gaussian_solver.evaluate()

    print("Variational dropout")
    variational_solver = Solver(dropout_method='variational')
    variational_solver.train(n_epochs=50)
    variational_solver.evaluate()