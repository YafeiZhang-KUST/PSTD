import torch
from torch import nn
import torch.nn.functional as F
#from pylab import *
import numpy as np

class LaplaceAlogrithm(nn.Module):
    def __init__(self):
        super(LaplaceAlogrithm, self).__init__()

    def forward(self, image):
        assert torch.is_tensor(image) is True

        laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)[np.newaxis, :, :].repeat(3, 0)
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).to(image.device)
        # laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0)#if no cuda
        image = image - F.conv2d(image, laplace_operator, padding=1, stride=1)
        return image

class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss, self).__init__()
        self.LaplaceAlogrithm = LaplaceAlogrithm()

    def forward(self, preds, labels):
        grad_img1 = self.LaplaceAlogrithm(preds)
        gt = self.LaplaceAlogrithm(labels)
        gt.requires_grad_(False)
        g_loss = F.l1_loss(grad_img1, gt, size_average=True, reduce=True)
        return g_loss



#class multiloss(nn.Module):
#    def __init__(self):
#        super(multiloss, self).__init__()
#        self.multiloss = HybridLoss()
#
#    def forward(self, preds_cartoon, preds_texture, preds, labels_texture, labels):
#        loss_cartoon = self.multiloss(preds_cartoon, labels)
#        loss_texture = self.multiloss(preds_texture, labels_texture)
#        # loss_compose = self.multiloss(preds_com, labels)
#        loss = self.multiloss(preds, labels)
#
#        loss = loss_cartoon + loss_texture + loss
#        return loss
        # return image_loss


class multiloss(nn.Module):
     def __init__(self):
         super(multiloss, self).__init__()
         self.l1loss = nn.L1Loss()
         self.multiloss = HybridLoss()
     def forward(self, preds_cartoon, preds_texture, preds, labels_texture, labels):
#         loss_cartoon =   self.l1loss(preds_cartoon, labels)
         loss_texture =   self.l1loss(preds_texture, labels_texture)
         # loss_compose = self.multiloss(preds_com, labels)
         loss = self.multiloss(preds, labels)

         loss = 0.001*loss_texture + loss
         return loss
        # return image_loss





class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.gr_loss = gradientloss()

    def forward(self, preds, labels):
        # Gradient loss
        gradientloss = self.gr_loss(preds, labels)
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - labels)
        # Image Loss
        image_loss = self.l1loss(preds, labels)
        image_loss1 = self.mse_loss(preds, labels)
        # TV Loss
        tv_loss = self.tv_loss(preds)
        return image_loss + 0.01*gradientloss + 0.000*image_loss1 + 0.001 * adversarial_loss + 0* tv_loss
        # return image_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



if __name__ == "__main__":
    g_loss = HybridLoss()
    print(g_loss)
