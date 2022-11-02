import scanpy as sc
import torch
import numpy as np
import random 

from torch.utils.data import DataLoader
import torch.autograd as autograd
from torch.autograd import Variable

from model import Encoder,Decoder,Discriminator,Discriminator2
from dataset import ScDataset

from utils import setup_seed
from utils import weights_init_normal
from tqdm import tqdm
from torch.distributions import Normal
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of {}' parameters: {}".format(name,num_params))

def calculate_gradient_penalty(real_data, fake_data, D):
    eta = torch.FloatTensor(real_data.size(0),1).uniform_(0,1)
    eta = eta.expand(real_data.size(0), real_data.size(1))
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        eta = eta.cuda()
    else:
        eta = eta

    interpolated = eta * real_data + ((1 - eta) * fake_data)

    if cuda:
        interpolated = interpolated.cuda()
    else:
        interpolated = interpolated

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


def train(scd, n_epochs, batch_size, n_critic):
    data_size = scd.adata.X.shape[1]
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    latent_dim = 256


    dataloader = DataLoader(
        dataset = scd,
        batch_size = batch_size,
        shuffle = True
    )

    # Initialize generator and discriminator
    encoder = Encoder(data_size,latent_dim)
    decoder = Decoder(data_size,latent_dim)
    discriminator = Discriminator(latent_dim)
    discriminator2 = Discriminator2(data_size)
    
    print_network(encoder, 'Encoder')
    print_network(decoder, 'Decoder')
    print_network(discriminator, 'Discriminator')
    print_network(discriminator2, 'Discriminator2')

    mse_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCELoss()
    
    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        discriminator2.cuda()
        mse_loss.cuda()
        bce_loss.cuda()

    # Initialize weights
    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(b1, b2))
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(b1, b2))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_discriminator2 = torch.optim.Adam(discriminator2.parameters(), lr=lr, betas=(b1, b2))
    
    loop = tqdm(range(n_epochs))
    for epoch in loop:
        encoder.train()
        decoder.train()
        for i, (data_A, data_B) in enumerate(dataloader):

            
            if  epoch % 2 ==0:
            # ---------------------
            #  Train autoencoder
            # ---------------------       

                real_data = Variable(data_B.type(FloatTensor))
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                x = Variable(data_A.type(FloatTensor))
                z, mu, var = encoder(x)
                gen_data = decoder(z)

                # loss_AE = mse_loss(gen_data, real_data) * real_data.size(-1)
                loss_AE = mse_loss(gen_data, real_data) * real_data.size(-1)
                loss_AE.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()

            # ---------------------
            #  Train Discriminator1
            # ---------------------
            
            x = Variable(data_A.type(FloatTensor))
            z, mu, var = encoder(x)
            gen_data = decoder(z)

            z_real = Variable(torch.randn(data_A.size(0), latent_dim).type(FloatTensor))
            optimizer_discriminator.zero_grad()
            z_fake = Variable(Normal(torch.zeros_like(mu),torch.ones_like(var)).sample().type(FloatTensor))

            # Loss for real images
            z_real_validity  = discriminator(z_real)
            z_fake_validity  = discriminator(z_fake)

            # Compute W-div gradient penalty
            z_div_gp = calculate_gradient_penalty(z_real, z_fake, discriminator)

            # Adversarial loss
            dz_loss = -torch.mean(z_real_validity) + torch.mean(z_fake_validity) + 10 * z_div_gp
            dz_loss.backward()
            optimizer_discriminator.step()

            # -----------------
            #  Train Generator1
            # -----------------

            if i % n_critic == 0:
    
                optimizer_encoder.zero_grad()
                x = Variable(data_A.type(FloatTensor), requires_grad=True)
                z, mu, var = encoder(x)
                z_fake = Variable(Normal(torch.zeros_like(mu),torch.ones_like(var)).sample().type(FloatTensor))
                
                z_fake_validity = discriminator(z_fake)
                gz_loss = -torch.mean(z_fake_validity)
                gz_loss.backward()
                optimizer_encoder.step()

                
                

            # ---------------------
            #  Train Discriminator2
            # ---------------------

            x = Variable(data_A.type(FloatTensor))
            z, mu, var = encoder(x)
            gen_data = decoder(z)
            optimizer_discriminator2.zero_grad()
            real_data = Variable(data_B.type(FloatTensor))

            # Loss for real images
            real_validity  = discriminator2(real_data)
            fake_validity  = discriminator2(gen_data)

            # Compute W-div gradient penalty
            div_gp = calculate_gradient_penalty(real_data, gen_data, discriminator2)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10*div_gp
            d_loss.backward()
            optimizer_discriminator2.step()



            # -----------------
            #  Train Generator2
            # -----------------

            if i % 100 == 0:

                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                x = Variable(data_A.type(FloatTensor), requires_grad=True)
                z, mu, var = encoder(x)
                gen_data = decoder(z)
                fake_validity = discriminator2(gen_data)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()


        # --------------
        # Log Progress
        # --------------
        loop.set_description(f'Epoch [{epoch}/{n_epochs}]')
        loop.set_postfix(Dz_loss=dz_loss.item(),Gz_loss=gz_loss.item(),D_loss=d_loss.item(),G_loss=g_loss.item())
        


    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        x = Variable(FloatTensor(scd.adata.X))
        z, mu, var = encoder(x)
        static_sample = decoder(z)
        latent_data = z.cpu().detach().numpy()
        fake_data = static_sample.cpu().detach().numpy()
    return fake_data,latent_data


def scAAE(ppd_adata, batch_key='batch', n_epochs=150,batch_size=1024, n_critic=100, seed=8):

    if seed is not None:
        setup_seed(seed)

    scd = ScDataset(ppd_adata)
    res_data,z = train(scd, n_epochs, batch_size, n_critic)

    return res_data,z
