import torch
import torch.nn as nn
class SymmetryLoss(nn.Module):

    def __init__(self, cross_entropy_loss, alpha=1.0):
        super().__init__()
        self.cross_entropy_loss = cross_entropy_loss
        self.mse_loss = nn.MSELoss(reduction='none')
        self.alpha = alpha

    def __call__(self, output, target):

        assert isinstance(output, dict)
        model_output = output['model_output']
        assert (model_output.shape[0] % 2) == 0

        auxiliary_output = output['auxiliary_output']
        assert (auxiliary_output.shape[0] % 2) == 0

        loss = self.cross_entropy_loss(model_output, target)

        output1 = auxiliary_output[0::2, ...]
        output2 = auxiliary_output[1::2, ...]

        target1 = target[0::2, ...]
        target2 = target[1::2, ...]
        diff = (target1 - target2).abs().flatten().sum()
        assert diff < 1e-6, 'Targets need to be the same'

        symmetry_loss = self.mse_loss(output1, output2)
        symmetry_loss = symmetry_loss.mean()

        loss += self.alpha * symmetry_loss

        return loss