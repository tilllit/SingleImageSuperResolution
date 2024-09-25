import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
import skimage
from dataset import *
from tqdm import tqdm


scale = 4
batch_size = 16

# Model path to test or resume
gen_path = 'model/SRGAN_gene_8000.pt'

#Training data
HR_TRpath = 'Dataset/DIV2K_train_HR'
LR_TRpath = 'Dataset/DIV2K_train_LR_bicubic/X4'




def train(epochs, pre_epochs=50, resume=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    print("Load data")
    transform = transforms.Compose([crop(scale, 24), augmentation()])
    dataset = mydata(GT_path = HR_TRpath, LR_path = LR_TRpath, in_memory = True, transform = transform)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    print("Successfuly loaded data!")

    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16, scale=scale)
    
    if resume:        
        generator.load_state_dict(torch.load(gen_path))
        print("pre-trained model is loaded")
        print("path : %s"%(gen_path))
        
    generator = generator.to(device)
    generator.train()
    
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr = 1e-4)
        
    pre_epoch = 0
    fine_epoch = 0
    
    #### Train using L2_loss
    print("Pretraining")
    while pre_epoch < pre_epochs:
        loop = tqdm(enumerate(loader), leave=True, total=len(loader), colour="GREEN")
        for i, tr_data in loop:
            loop.set_description(f"Epoch [{pre_epoch + 1}/{pre_epochs}]")
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

            loop.set_postfix(loss=loss.item())

        pre_epoch += 1

    torch.save(generator.state_dict(), './model/pre_trained_model_%03d.pt'%pre_epoch)

        
    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()
    
    discriminator = Discriminator(patch_size = 24 * scale)
    discriminator = discriminator.to(device)
    discriminator.train()
    
    d_optim = optim.Adam(discriminator.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size = 2000, gamma = 0.1)
    
    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    real_label = torch.ones((batch_size, 1)).to(device)
    fake_label = torch.zeros((batch_size, 1)).to(device)
    
    print("Training GAN")
    while fine_epoch < epochs:
        
        scheduler.step()
        
        loop = tqdm(enumerate(loader), leave=True, total=len(loader), colour="GREEN")
        for i, tr_data in loop:
            loop.set_description(f"Epoch [{fine_epoch + 1}/{epochs}]")
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)
                        
            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            
            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            
            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer = 'relu5_4')
            
            L2_loss = l2_loss(output, gt)
            percep_loss = 0.006 * _percep_loss
            adversarial_loss = 1e-3 * cross_ent(fake_prob, real_label)
            total_variance_loss = 0.0 * tv_loss(0.006 * (hr_feat - sr_feat)**2)
            
            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss
            
            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            loop.set_postfix(loss=g_loss.item())

            
        fine_epoch += 1

    torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
    torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)


def load_image(path, downfactor=4):
    #original_img = load_img(path)
    original_img = Image.open(path).convert("RGB")
    downscaled = original_img.resize((original_img.size[0] // downfactor,
                                      original_img.size[1] // downfactor), Image.BICUBIC)
    
    downscaled = np.array(downscaled).astype(np.uint8)

    downscaled = (downscaled / 127.5) - 1.0
    downscaled = downscaled.transpose(2, 0, 1).astype(np.float32)
    
    return original_img, downscaled


def test():

    test_path = '0003.png'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16)
    generator.load_state_dict(torch.load(gen_path, map_location=torch.device(device)))
    generator = generator.to(device)
    generator.eval()

    original_img, downscaled = load_image(test_path)
    downscaled = torch.from_numpy(np.asarray([downscaled]))

    output, _ = generator(downscaled)
    output = output[0].detach().numpy()
    output = np.clip(output, -1.0, 1.0)
    output = (output + 1.0) / 2.0
    output = output.transpose(1,2,0)

    result = Image.fromarray((output * 255.0).astype(np.uint8))
    result.save('gan.png')


def main():
    #train(epochs=10, pre_epochs=1, resume=False)
    test()


if __name__ == '__main__':
    main()

