import torch
import torch.nn as nn


class GradientLoss(nn.Module):
    def __init__(self, dx=1):
        super(GradientLoss, self).__init__()
        self.dx = dx

    def forward(self,
                predict: torch.tensor,
                targets: torch.tensor):

        dudx_pred = predict[:, 0, 2:, :, :] - predict[:, 0, :-2, :, :]
        dvdy_pred = predict[:, 1, :, 2:, :] - predict[:, 1, :, :-2, :]
        dwdz_pred = predict[:, 2, :, :, 2:] - predict[:, 2, :, :, :-2]

        dudx_targ = targets[:, 0, 2:, :, :] - targets[:, 0, :-2, :, :]
        dvdy_targ = targets[:, 1, :, 2:, :] - targets[:, 1, :, :-2, :]
        dwdz_targ = targets[:, 2, :, :, 2:] - targets[:, 2, :, :, :-2]

        shape = predict.shape

        loss = (torch.sum(torch.square((dudx_pred-dudx_targ)/self.dx)) +
                torch.sum(torch.square((dvdy_pred-dvdy_targ)/self.dx)) +
                torch.sum(torch.square((dwdz_pred-dwdz_targ)/self.dx))) / (shape[2]*shape[3]*shape[4])

        return loss