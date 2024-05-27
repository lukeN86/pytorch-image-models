import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class SymmetryLoss(nn.Module):

    def __init__(self, cross_entropy_loss, symmetry_regularization='l2', alpha=1.0, symmetry_transformation=None):
        super().__init__()
        self.cross_entropy_loss = cross_entropy_loss

        if symmetry_regularization == 'l2':
            self.symmetry_regularization_loss = nn.MSELoss(reduction='none')
        elif symmetry_regularization == 'smooth_l1':
            self.symmetry_regularization_loss = nn.SmoothL1Loss(reduction='none')
        elif symmetry_regularization == 'l1':
            self.symmetry_regularization_loss = nn.L1Loss(reduction='none')
        else:
            assert False, 'Unknown symmetry regularization'

        self.alpha = alpha
        self.symmetry_transformation=symmetry_transformation

    def __call__(self, output, target):

        assert isinstance(output, dict)
        model_output = output['model_output']
        assert (model_output.shape[0] % 2) == 0

        auxiliary_output = output['auxiliary_output']
        assert (auxiliary_output.shape[0] % 2) == 0

        if isinstance(target, dict):
            labels = target['labels']
        else:
            assert isinstance(target, torch.Tensor)
            labels = target

        loss = self.cross_entropy_loss(model_output, labels)

        assert not loss.isnan(), 'Invalid cross entropy loss'

        symmetry_loss_mask = None
        if self.symmetry_transformation is not None:
            if self.symmetry_transformation == 'augmentation_equivariance':
                inv_augmentation_transform = target['inv_augmentation_transform']

                symmetry_loss_mask1 = inv_augmentation_transform[0::2, :, :, 0] > -5
                symmetry_loss_mask2 = inv_augmentation_transform[1::2, :, :, 0] > -5

                symmetry_loss_mask = torch.logical_and(symmetry_loss_mask1, symmetry_loss_mask2)
                symmetry_loss_mask = symmetry_loss_mask[:, None, :, :].repeat(1, auxiliary_output.size(1), 1, 1).float()
                assert symmetry_loss_mask.sum() > 0
                assert symmetry_loss_mask.sum() > 10

                auxiliary_output = F.grid_sample(auxiliary_output, inv_augmentation_transform, align_corners=True)
            else:
                assert False, 'Unknown symmetry transformation {}'.format(self.symmetry_transformation)

        output1 = auxiliary_output[0::2, ...]
        output2 = auxiliary_output[1::2, ...]

        labels1 = labels[0::2, ...]
        labels2 = labels[1::2, ...]
        diff = (labels1 - labels2).abs().flatten().sum()
        assert diff < 1e-6, 'Targets need to be the same'

        symmetry_loss = self.symmetry_regularization_loss(output1, output2)

        if symmetry_loss_mask is not None:
            symmetry_loss *= symmetry_loss_mask

        symmetry_loss = symmetry_loss.mean()

        assert not symmetry_loss.isnan(), 'Invalid symmetry loss'

        loss += self.alpha * symmetry_loss

        return loss