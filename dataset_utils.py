import torch


def dataset_mean_and_var(dataset):
    """Compute the mean and standard deviation of a torchvision dataset


    Parameters:
    ----------
    dataset : {torchvision.datasets}
        The dataset
    
    Returns
    -------
    tuple
        Mean and variance of the dataset channels
    
    Examples
    --------
        >> from torchvision import datasets, transforms
        >> dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        >> mean, std = dataset_mean_and_var(dataset)
        >> dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))

        >> from torchvision import datasets, transforms
        >> dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
        >> mean, std = dataset_mean_and_var(dataset)
        >> dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    """

    def parallel_variance(avg_a, var_a, count_a, avg_b, var_b, count_b):
        """Method for online updating an sample mean and variance given a new set of observations.
        
        See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        
        Parameters:
        ----------
        avg_a : {float}
            The current sample mean.
        var_a : {float}
            The current sample variance
        count_a : {int}
            The number of observations in the current sample mean.
        avg_b : {float}
            The sample mean of the new set of observations.
        var_b : {float}
            The sample variance of the new set of observations.
        count_b : {int}
            The number of observations in the new set.

        Returns
        -------
        tuple
            The sample mean and variance of the total sample
        """
        count_x = count_a + count_b
        delta = avg_b - avg_a
        M2_a = var_a * (count_a - 1)
        M2_b = var_b * (count_b - 1)
        M2_x = M2_a + M2_b + delta ** 2 * count_a * count_b / count_x
        var_x = M2_x / (count_a + count_b - 1)
        avg_x = count_a / count_x * avg_a + count_b / count_x * avg_b
        return avg_x, var_x

    # Image data
    if dataset.__class__ in [MNIST, FashionMNIST, CIFAR10, CIFAR100]:
        # Get example dimensions
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=torch.multiprocessing.cpu_count())
        d, t = next(iter(dataloader))
        batch_size, n_channels, height, width = d.size()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=torch.multiprocessing.cpu_count())
        # Compute
        mean, var = torch.zeros(n_channels), torch.zeros(n_channels)
        n_per_example = height * width
        n_total = 0
        for n, (d, t) in enumerate(dataloader):
            d = d.view(n_channels, height, width)
            for i in range(n_channels):
                mean[i], var[i] = parallel_variance(mean[i], var[i], n_total, d[i, ...].mean(), d[i, ...].var(), d[i, ...].numel())
                n_total += n_per_example
            if n % (len(dataloader) // 100) == 0:
                print(str(n) + "/" + str(len(dataloader)))
        return mean, var
    else:
        raise NotImplementedError('The {} dataset is not implemeted here yet'.format(dataset))


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
    print(avg, std)
