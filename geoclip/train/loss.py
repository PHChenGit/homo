from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from geoclip.model.geotnf.point_tnf import PointTnf


def orientation_loss(pred, target):
    # cos_theta = pred[:, 1]
    # sin_theta = pred[:, 0]
    # angle_rad = torch.atan2(sin_theta, cos_theta)
    # pred = torch.rad2deg(angle_rad)

    N = torch.divide(pred, 360)
    pred = torch.where(pred >= 360, pred - (N * 360), pred)
    pred = torch.where(pred < 0, pred + (torch.abs(N) * 360), pred)
    
    error1 = torch.abs(target - pred)
    error2 = torch.sub(360, error1)
    error = torch.min(error1, error2)
    mae = torch.mean(error)
    return mae, error

def orientation_rmse(pred, target):
    # N = torch.divide(pred, 360)
    # pred = torch.where(pred >= 360, pred - (N * 360), pred)
    # pred = torch.where(pred < 0, pred + (torch.abs(N) * 360), pred)
    
    error1 = torch.abs(target - pred)
    error2 = torch.sub(360, error1)
    error = torch.min(error1, error2)
    mae = torch.mean(error)
    return mae, error

class TransformedGridLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda();

    def forward(self, theta, theta_GT):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)
        # compute transformed grid points using estimated and GT tnfs

        # if self.geometric_model=='affine':
        #     P_prime = self.pointTnf.affPointTnf(theta,P)
        #     P_prime_GT = self.pointTnf.affPointTnf(theta_GT,P)
        # elif self.geometric_model=='hom':
        #     P_prime = self.pointTnf.homPointTnf(theta,P)
        #     P_prime_GT = self.pointTnf.homPointTnf(theta_GT,P)
        # elif self.geometric_model=='tps':
        #     P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).unsqueeze(3),P)
        #     P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT,P)

        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)

        return loss
