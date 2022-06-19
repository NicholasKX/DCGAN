# -*- coding: utf-8 -*-
"""
Created on 2022/6/19 10:41 
@Author: Wu Kaixuan
@File  : vae.py 
@Desc  : vae 
"""
import torch
import torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=3,
                              padding=1,
                              stride=2,)
        self.bn = nn.BatchNorm2d(num_features=out_channel,momentum=0.9)
        self.relu = nn.ReLU()
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x3 = self.relu(x2)
        return x3

class VAE_Encoder(nn.Module):
    def __init__(self,in_channel=3,latent_z=128):
        super(VAE_Encoder, self).__init__()
        self.in_channel=in_channel
        layers_list=[]
        for i in range(5):
            if i==0:
                layers_list.append(EncoderBlock(self.in_channel,
                                                out_channel=64))
                self.in_channel=64
            else:
                layers_list.append(EncoderBlock(self.in_channel,
                                                out_channel=self.in_channel*2))
                self.in_channel*=2
        self.layers=nn.Sequential(*layers_list)
        self.fc = nn.Sequential(
            nn.Linear(in_features=2*2*self.in_channel,out_features=1024),
            nn.BatchNorm1d(1024,momentum=0.9),
            nn.ReLU(True),
        )
        self.l_mu = nn.Linear(in_features=1024,out_features=latent_z)
        self.l_var = nn.Linear(in_features=1024,out_features=latent_z)

    def forward(self,x):
        x = self.layers(x)
        x = x.view(len(x),-1)
        x = self.fc(x)
        mu = self.l_mu(x)
        var = self.l_var(x)
        return mu,var

if __name__ == '__main__':
    encoder = VAE_Encoder()
    print(encoder)
    input = torch.randn(2,3,64,64)
    output = encoder(input)
    print(output)
