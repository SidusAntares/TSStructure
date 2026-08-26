import torch.nn as nn
from models.layers import LinearLayer


def get_decoder(n_neurons, n_classes):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu

    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
        n_classes (int): Output size
    """
    layers = []
    for i in range(len(n_neurons) - 1):
        layers.append(LinearLayer(n_neurons[i], n_neurons[i + 1]))
    layers.append(nn.Linear(n_neurons[-1], n_classes))
    m = nn.Sequential(*layers)
    return m


class MTKDLateLogitDecoder(nn.Module):
    """Apply independent classifiers to packed T/S features and sum raw logits."""

    def __init__(self, n_neurons, num_classes):
        super().__init__()
        self.branch_dim = n_neurons[0]
        self.classifier_t = get_decoder(list(n_neurons), num_classes)
        self.classifier_s = get_decoder(list(n_neurons), num_classes)

    def forward(self, temporal_feats):
        if temporal_feats.shape[-1] != 2 * self.branch_dim:
            raise ValueError(
                "temporal_feats must pack [z_T || z_S] with size {}".format(
                    2 * self.branch_dim
                )
            )
        z_t = temporal_feats[:, : self.branch_dim]
        z_s = temporal_feats[:, self.branch_dim :]
        logits_t = self.classifier_t(z_t)
        logits_s = self.classifier_s(z_s)
        return logits_t + logits_s
