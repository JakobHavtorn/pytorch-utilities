from collections import OrderedDict

import IPython
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable


def get_names_dict(model):
    """
    Recursive walk over modules to get names including path.
    """
    names = {}

    def _get_names(module, parent_name=''):
        for key, module in module.named_children():
            name = parent_name + '.' + key if parent_name else key
            names[name] = module
            if isinstance(module, torch.nn.Module):
                _get_names(module, parent_name=name)
    _get_names(model)
    return names


def summarize_model(model, input_size, return_meta=False):
    """Summarizes torch model by showing trainable parameters and weights.

    Parameters:
    ----------
    model : {nn.Module}
        The model to summarize.
    input_size : {tuple}
        The dimensions of the model input not including batch size.
    return_meta : {bool}, optional
        Whether or not to return some additional meta data of the 
        model compute from the summary (the default is False).

    Returns
    -------
    pd.DataFrame
        The model summary as a Pandas data frame.

    ---------
    Example:
        import torchvision.models as models
        model = models.alexnet()
        df = summarize_model(model=model, input_size=(3, 224, 224))
        print(df)

                     name class_name        input_shape       output_shape  n_parameters
        1     features=>0     Conv2d  (-1, 3, 224, 224)   (-1, 64, 55, 55)      23296
        2     features=>1       ReLU   (-1, 64, 55, 55)   (-1, 64, 55, 55)          0
        ...
    """
    def get_settings(m):
        c = m.__class__
        s = {}
        # Linear layers
        if c is [nn.Linear, nn.Bilinear]:
            s = '-'
        # Convolutional layers
        if c in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            s = {'stride': m.stride, 'padding': m.padding}
        if c in [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
            s = {'stride': m.stride, 'padding': m.padding, 'output_padding': m.output_padding}
        # Pooling layers
        if c in [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]:
            s = {'kernel_size': m.kernel_size, 'stride': m.stride, 'padding': m.padding, 'dilation': m.dilation} #, 'ceil_mode'=False}
        if c in [nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d]:
            s = {'kernel_size': m.kernel_size, 'stride': m.stride, 'padding': m.padding}
        if c in [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]:
            s = {'kernel_size': m.kernel_size, 'stride': m.stride, 'padding': m.padding, 'count_include_pad': m.count_include_pad}
        # Padding layers
        if c in [nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d, 
                 nn.ZeroPad2d, nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]:
            s = {'padding': m.padding}
            if c in [nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]:
                s['value'] = m.value
        # Recurrent layers
        if c in [nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell]:
            s = {'input_size': m.input_size, 'hidden_size': m.hidden_size,
                 'num_layers': m.num_layers, 'nonlinearity': m.nonlinearity,
                 'dropout': m.dropout, 'bidirectional': m.bidirectional,
                 'batch_first': m.batch_first}
        # Dropout layers
        if c in [nn.Dropout, nn.Dropout2d, nn.Dropout3d]:
            s = {'p': m.p}
        # Normalization layers
        if c in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
            s = {'momentum': m.momentum, 'affine': m.affine}
        # Activation functions
        # Embedding layers
        s = s if len(s) > 0 else '-'
        return s

    def register_hook(module):
        # Define hook
        def hook(module, input, output):
            name = ''
            for key, item in names.items():
                if item == module:
                    name = key
            # Get class name and set module index
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = module_idx + 1
            # Prepare summary entry for this module
            summary[m_key] = OrderedDict()
            summary[m_key]['name'] = name
            summary[m_key]['class_name'] = class_name
            # Input and output shape
            summary[m_key]['input_shape'] = (-1, ) + tuple(input[0].size())[1:]
            summary[m_key]['output_shape'] = (-1, ) + tuple(output.size())[1:]
            # Weight dimensions
            summary[m_key]['weight_shapes'] = list([tuple(p.size()) for p in module.parameters()])
            # Number of parameters in layers
            summary[m_key]['n_parameters'] = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters()])            
            summary[m_key]['n_trainable'] = sum([torch.LongTensor(list(p.size())).prod() for p in module.parameters() if p.requires_grad])
            # Get special settings for layers
            summary[m_key]['settings'] = get_settings(module)

        # Append
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Put model in evaluation mode (required for some modules {BN, DO, etc.})
    was_training = model.training
    if model.training:
        model.eval()
    # Names are stored in parent and path+name is unique not the name
    names = get_names_dict(model)
    # Check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))
    # Move parameters to CUDA if relevant
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    # Create properties
    summary = OrderedDict()
    hooks = []
    # Register hook on all modules of model
    model.apply(register_hook)
    # Make a forward pass to evaluate registered hook functions
    # and build summary
    model(x)
    # Remove all the registered hooks from the model again and
    # return it in the state it was given.
    for h in hooks:
        h.remove()
    # If the model was in training mode, put it back into training mode
    if was_training:
        model.train()
    # Make dataframe
    df_summary = pd.DataFrame.from_dict(summary, orient='index')
    # Create additional info
    if return_meta:
        meta = {'total_parameters': df_summary.n_parameters.sum(),
                'total_trainable': df_summary.n_trainable.sum(),
                'layers': df_summary.shape[0],
                'trainable_layers': (df_summary.n_trainable != 0).sum()}
        df_meta = pd.DataFrame.from_dict(meta, orient='index')
        return df_summary, df_meta
    else:
        return df_summary


if __name__ == '__main__':
    import torchvision.models as models
    pd.set_option('display.max_colwidth', -1)
    # AlexNet, ResNet-152 and VGG19 with BN
    models = [models.alexnet(), models.resnet152(), models.vgg19_bn()]
    for model in models:
        df, meta = summarize_model(model=model, input_size=(3, 224, 224), return_meta=True)
        print(df.to_string())
        print(meta)
        print('\n##################################################################\n')
