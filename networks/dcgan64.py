# -*- coding: utf-8 -*-
"""
Created on 2022/5/18 22:51 
@Author: Wu Kaixuan
@File  : dcgan64.py 
@Desc  : dcgan64 
"""
import torch.nn as nn
import torch
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, output_size=64,latent_dim=100,out_channels=3):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_dim, output_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(output_size * 8),
            nn.ReLU(True),
            # state size. (output_size*8) x 4 x 4
            nn.ConvTranspose2d(output_size * 8, output_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size * 4),
            nn.ReLU(True),
            # state size. (output_size*4) x 8 x 8
            nn.ConvTranspose2d( output_size * 4, output_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size * 2),
            nn.ReLU(True),
            # state size. (output_size*2) x 16 x 16
            nn.ConvTranspose2d( output_size * 2, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),
            # state size. (output_size) x 32 x 32
            nn.ConvTranspose2d( output_size, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3,input_size=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels, input_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size) x 32 x 32
            nn.Conv2d(input_size, input_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size*2) x 16 x 16
            nn.Conv2d(input_size * 2, input_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size*4) x 8 x 8
            nn.Conv2d(input_size * 4, input_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size*8) x 4 x 4
            nn.Conv2d(input_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':
    gen=Generator()
    input = torch.randn(64,100,1,1)
    output = gen(input)
    dis = Discriminator()
    cls = dis(output)
    print(output.shape)
    print(cls.shape)
