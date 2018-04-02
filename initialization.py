from torch import nn


def calculate_xavier_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function. The values are as follows:
    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    Linear       :math:`1`
    Conv{1,2,3}d :math:`1`
    Sigmoid      :math:`1`
    Tanh         :math:`5 / 3`
    ReLU         :math:`\sqrt{2}`
    Leaky_ReLU   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn` name, i.e. module name)
        param: optional parameter for the nonlinear function

    Examples:
        >>> gain = calculate_xavier_gain(nn.modules.ReLU)
    """
    linear_fns = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d , nn.ConvTranspose2d , nn.ConvTranspose3d]
    islinear = any([nonlinearity.__name__ == lf.__name__ for lf in linear_fns])
    if islinear or nonlinearity.__name__ == nn.Sigmoid.__name__:
        return 1
    elif nonlinearity.__name__ == nn.Tanh.__name__:
        return 5.0 / 3
    elif nonlinearity.__name__ == nn.ReLU.__name__:
        return math.sqrt(2.0)
    elif nonlinearity.__name__ == nn.LeakyReLU.__name__:
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
