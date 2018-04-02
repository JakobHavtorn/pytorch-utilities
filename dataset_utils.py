import torch


def compute_avg_and_std(dataset, dim=0):
    """Compute the avg and standard deviation of a torchvision dataset

    Usage:
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    IPython.embed()
    avg = torch.zeros(3)
    std = torch.zeros(3)
    for inp, tgt in dataloader:
        for i in range(1):
            avg[i] += inp[:, i, :, :].avg()
            std[i] += inp[:, i, :, :].std()
    avg.div_(len(dataset))
    std.div_(len(dataset))
    return avg, std
