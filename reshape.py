import torch.nn as nn


class Flatten(nn.Module):
    """Flattens the batch.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class PrintBatch(nn.Module):
    """Prints the forward propagated batch.
    """
    def __init__(self):
        super(PrintBatch, self).__init__()

    def forward(self, x):
        print(self.__class__, x)
        return x
    

class PrintDim(nn.Module):
    """Prints the dimension of the forward propagated batch.
    """    
    def __init__(self):
        super(PrintDim, self).__init__()

    def forward(self, x):
        print(x.size())
        return x