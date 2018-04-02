import torch



def update(existingAggregate, newValue):
    """# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
count aggregates the number of samples seen so far
    
    Parameters:
    ----------
    existingAggregate : {tuple}
        The aggregated the mean, standard deviation and sum of 
        squares of differences from the current mean.
    newValue : {[type]}
        [description]
    Returns
    -------
    [type]
        [description]
    """
    (count, mean, M2) = existingAggregate
    count = count + 1 
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)

# retrieve the mean and variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance) = (mean, M2/(count - 1)) 
    if count < 2:
        return float('nan')
    else:
        return (mean, variance)


def compute_avg_and_std(dataset, dim=1):
    """Compute the avg and standard deviation of a torchvision dataset.
    
    Parameters:
    ----------
    dataset : {torchvision.datasets}
        The dataset
    dim : {int}, optional
        The dimension of an example from the dataset to .
        The possible dimensions are
            0:      Batch dimension (example id)
            1:      Data channel; Color channel for color images
            2:      Height dimension for images
            3:      Width dimension for images
            None:   Return a single statistic for the entire dataset (e.g. MNIST greyscale channel)
        (the default is 1)
    
    Returns
    -------
    tuple
        The mean and the standard deviation

    Example
    -------
        from torchvision import datasets, transforms
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        avg, std = compute_avg_and_std(dataset)
        dataset = datasets.MNIST(data_dir, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (avg,), (std,))
                                ]))

        from torchvision import datasets, transforms
        dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
        avg, std = compute_avg_and_std(dataset)
        dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (avg,), (std,))
                                ]))
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=torch.multiprocessing.cpu_count())
    d, _ = next(iter(dataloader))
    dim_size = d.size()[dim]
    avg = torch.zeros(dim_size)
    std = torch.zeros(dim_size)
    for d, _ in dataloader:
        for i in range(dim_size):
            avg[i] += d[:, i, :, :].mean()
            std[i] += d[:, i, :, :].std()
    avg.div_(len(dataset))
    std.div_(len(dataset))
    return avg, std


# Uses online estimation of mean and variance to reduce memory requirement.
# See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    
if __name__ == '__main__':
    import IPython
    from torchvision import datasets, transforms
    data_dir = './data'
    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
    avg, std = compute_avg_and_std(dataset)
    dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (avg,), (std,))
                        ]))
