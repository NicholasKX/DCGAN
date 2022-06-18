# -*- coding: utf-8 -*-
"""
Created on 2022/5/8 21:09 
@Author: Wu Kaixuan
@File  : dcgan.py
@Desc  : dagan 
"""
import torch
import torch.nn as nn

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self,input_size=512,latent_dim=100,output_channels=3):
        super(Generator, self).__init__()
        input_size = input_size//4
        self.model = nn.Sequential(
            # input is latent_dim, going into a convolution
            nn.ConvTranspose2d(latent_dim,input_size*64,4,1,0,bias=False),
            nn.BatchNorm2d(input_size*64),
            nn.ReLU(True),
            # state size. (input_size*8) x 4 x 4
            nn.ConvTranspose2d(input_size*64,input_size*32,4,2,1,bias=False),
            nn.BatchNorm2d(input_size*32),
            nn.ReLU(True),
            # state size. (input_size*4) x 8 x 8
            nn.ConvTranspose2d(input_size*32,input_size*16,4,2,1,bias=False),
            nn.BatchNorm2d(input_size*16),
            nn.ReLU(True),
            # state size. (input_size*2) x 16 x 16
            nn.ConvTranspose2d(input_size*16,input_size*8,4,2,1,bias=False),
            nn.BatchNorm2d(input_size*8),
            nn.ReLU(True),
            # state size. (input_size) x 32 x 32
            nn.ConvTranspose2d(input_size * 8, input_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size*4),
            nn.ReLU(True),
            # state size. (input_size) x 64 x 64
            nn.ConvTranspose2d(input_size * 4, input_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size*2),
            nn.ReLU(True),
            # state size. (input_size) x 128 x 128
            nn.ConvTranspose2d(input_size * 2, input_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size),
            nn.ReLU(True),
            # state size. (input_size) x 256 x 256
            nn.ConvTranspose2d(input_size,output_channels,4,2,1,bias=False),
            nn.Tanh(),
            # state size. (output_channels) x 512 x 512
        )

    def forward(self, input):
        output = self.model(input)
        return output

class Discriminator(nn.Module):
    def __init__(self,input_size=512,input_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False,dropout = 0):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if dropout:
                block.append(nn.Dropout2d(dropout))
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block
        input_size = input_size//8
        self.model = nn.Sequential(
            #3*512*512-->*256*256
            *discriminator_block(input_channels, input_size, bn=False),
            #*256*256-->*128*128
            *discriminator_block(input_size, input_size*2),
            # *128*128-->*64*64
            *discriminator_block(input_size*2, input_size*4),
            # *64*64-->*32*32
            *discriminator_block(input_size*4, input_size*8),
            # *32*32-->*16*16
            *discriminator_block(input_size*8, input_size*16),
            # *16*16-->*8*8
            *discriminator_block(input_size*16, input_size*32),
            # *8*8-->*4*4
            *discriminator_block(input_size*32, input_size*64),
        )
        self.adv_layer = nn.Sequential(nn.Conv2d(input_size*64,1,4,1,0,bias=False), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        validity = self.adv_layer(out)
        return validity


if __name__ == '__main__':
    import numpy as np
    gen = Generator(input_size=512,latent_dim=100)
    gen.apply(weights_init)
    dis = Discriminator(input_size=512,input_channels=3)
    # x = torch.Tensor(np.random.normal(0, 1, (1, 100)))
    # Generate batch of latent vectors
    noise = torch.randn(8, 100, 1, 1)
    output = gen(noise)
    y = dis(output)
    print(output.size())
    print(y.size())