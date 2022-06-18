# -*- coding: utf-8 -*-
"""
Created on 2022/5/18 23:11 
@Author: Wu Kaixuan
@File  : train_celeba.py 
@Desc  : train_celeba 
"""
# -*- coding: utf-8 -*-
"""
Created on 2022/5/10 21:10 
@Author: Wu Kaixuan
@File  : train_pix2pix.py 
@Desc  : train 
"""
import os
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from networks.dcgan64 import Generator,Discriminator,weights_init
from dataloader import LEVIR
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from IPython.display import HTML

# os.environ['CUDA_VISIBLE_DEVICES']='3'
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = '../datasets/img_align_celeba'
output_path = './model_weight/celeba_64'
if not os.path.exists(output_path):
    os.mkdir(output_path)
tmp_imgs = './tmp_imgs_64'
if not os.path.exists(tmp_imgs):
    os.mkdir(tmp_imgs)
batch_size = 128
num_works = 2
img_size = 64
input_channels=3
latent_dim =100
train_epochs = 10
lr = 0.0002
beta1 = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False
if __name__ == '__main__':
    dataset = LEVIR(root=dataroot,
                          transform=transforms.Compose([
                              transforms.Resize(img_size),
                              transforms.CenterCrop(img_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=num_works)
    writer = SummaryWriter()
    # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    # Loss function
    adversarial_loss = nn.BCELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.to(device)
        discriminator.to(device)
        adversarial_loss.to(device)
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    iters = 0
    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    img_list=[]
    for epoch in tqdm(range(train_epochs)):
        for i, imgs in enumerate(dataloader):
            b_size = batch_size
            valid = torch.Tensor(b_size, 1).fill_(1.0).to(device)
            fake = torch.Tensor(b_size, 1).fill_(0.0).to(device)
            real_imgs = imgs.to(device)
            #Updata D
            discriminator.zero_grad()
            output = discriminator(real_imgs).view(-1,1)
            # print(output.shape)
            if len(output)!=len(valid):
                valid = torch.Tensor(len(output), 1).fill_(1.0).to(device)
            errD_real = adversarial_loss(output,valid)
            # errD_real.backward()
            # D_x = output.mean().item()
            #利用噪声生成imgs
            noise = torch.randn(b_size,latent_dim,1,1,device=device)
            fake_imgs = generator(noise)
            output=discriminator(fake_imgs.detach()).view(-1,1)
            if len(output) != len(fake):
                fake = torch.Tensor(len(output), 1).fill_(0.0).to(device)
            errD_fake = adversarial_loss(output,fake)
            # errD_fake.backward()
            # D_G_z1 = output.mean().item()
            errD = (errD_real+errD_fake)/2
            errD.backward()
            optimizer_D.step()
            #update G
            generator.zero_grad()
            output = discriminator(fake_imgs).view(-1,1)
            if len(output) != len(valid):
                valid = torch.Tensor(len(output), 1).fill_(1.0).to(device)
            errG = adversarial_loss(output,valid)
            errG.backward()
            # D_G_z2 = output.mean().item()
            optimizer_G.step()
            writer.add_scalar(tag='D_loss',scalar_value=errD.item(),global_step=i)
            writer.add_scalar(tag='G_loss',scalar_value=errG.item(),global_step=i)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, train_epochs, i, len(dataloader), errD.item(), errG.item())
            )

            if (iters % 500 == 0) or ((epoch == train_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_img = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_img,nrow=8, padding=2, normalize=True))

            iters += 1


        torch.save(generator.state_dict(), os.path.join(output_path, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(output_path, f'discriminator_epoch_{epoch}.pth'))
        print('Save Model Succeed')
        writer.close()

        fig = plt.figure()
        # plt.axis("off")
        # ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        #
        # HTML(ani.to_jshtml())

        plt.axis("off")
        for i in img_list:
            plt.imshow(np.transpose(i, (1, 2, 0)),animated=True)
        plt.savefig(os.path.join(tmp_imgs,f'{iters}.png'))
