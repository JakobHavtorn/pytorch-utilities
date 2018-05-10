from model_summary import summarize_model
from initialization import calculate_xavier_gain
import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    """Abstract models class for pytorch models.
    
    Adds methods for counting parameters, tensors and layers.
    Adds summary property for getting summarizing statistics and model layout
    """
    @property
    def summary(self):
        if not hasattr(self, '_summary'):
            self._summary = summarize_model(self, self.in_dim)
        return self._summary

    def count_parameters(self, only_trainable=True):
        """Return the number of [trainable] parameters in this model.
        """
        return self._count_parameters(self, only_trainable=only_trainable)

    @staticmethod
    def _count_parameters(m, only_trainable=True):
        """Count the number of [trainable] parameters in a pytorch model.
        """
        k = 'n_trainable' if only_trainable else 'n_parameters'
        return int(m.summary[k].sum())

    def count_tensors(self, only_trainable=True):
        return self._count_tensors(self, only_trainable=only_trainable)

    @staticmethod
    def _count_tensors(m, only_trainable=True):
        """Count the number of [trainable] tensor objects in a pytorch model.
        """
        k = 'n_trainable' if only_trainable else 'n_parameters'
        return sum([1 for i, l in m.summary.iterrows() for w in l['weight_shapes'] if l['weight_shapes'] and l[k] > 0])

    def count_layers(self, only_trainable=True):
        """Count the number of [trainable] layers in a pytorch model.
        A layer is defined as a module with a nonzero number of [trainable] parameters.
        """
        return self._count_layers(self, only_trainable=only_trainable)
    
    @staticmethod
    def _count_layers(m, only_trainable=True):
        k = 'n_trainable' if only_trainable else 'n_parameters'
        return m.summary[m.summary[k] > 0].shape[0]

    def initialize_weights(self):
        def init_gru(cell, gain=1):
            # Orthogonal initialization of recurrent weights
            cell.reset_parameters()
            for _, hh, _, _ in cell.all_weights:
                for i in range(0, hh.size(0), cell.hidden_size):
                    I.orthogonal(hh[i:i + cell.hidden_size], gain=gain)
        
        def init_lstm(cell, gain=1):
            # Orthogonal initialization of recurrent weights and 
            # positive forget gate bias (Jozefowicz et al., 2015)
            init_gru(cell, gain)
            for _, _, ih_b, hh_b in cell.all_weights:
                l = len(ih_b)
                ih_b[l // 4:l // 2].data.fill_(1.0)
                hh_b[l // 4:l // 2].data.fill_(1.0)
    
        # Loop in reverse to pick up the nonlinearity following the layer for gain computation
        modules = list(self.modules())
        for m in reversed(modules):
            try:
                gain = calculate_xavier_gain(m.__class__)
            except:
                gain = 1
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                assert gain == calculate_xavier_gain(nn.Conv1d)
                nn.init.xavier_normal(m.weight.data, gain=gain)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                init_lstm(m, gain=gain)
            elif isinstance(m, nn.GRU):
                init_gru(m, gain=gain)
            elif isinstance(m, nn.Linear):
                assert gain == calculate_xavier_gain(nn.Linear)
                nn.init.xavier_normal(m.weight.data, gain=gain)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if m.affine:
                    # Affine transform does nothing at first
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                # if m.track_running_stats:
                # Running stats are initialized to have no history
                m.running_mean.zero_()
                m.running_var.fill_(1)

    
