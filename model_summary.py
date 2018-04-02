from collections import OrderedDict

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable


def get_names_dict(model):
    """Recursive walk to get names of modules including path for a pytorch model.
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

        # Append 
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Put model in evaluation mode (required for e.g. batchnorm layers)
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
