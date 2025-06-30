import torch.nn as nn
import torch, argparse

import os,sys
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
np.random.seed(0)
torch.manual_seed(0)
save_base = '../../save'
epochs = 1000
batch_size = 1
learning_rate = 5e-4

using_kernel_size = 3
epoch_check = 1
tag = 'torch_20250620_unet_relu_adam_wormearth20new'

device = 'cuda:0'

def load_data():
    train_label = torchvision.io.read_image('../../data/earth.jpg') # torch.Size([3, 2160, 3840])
    h,w = train_label.shape[-2:]
    train_label = train_label.reshape(1,3,h,w).type(torch.float32).to(device) / 255
    train_inp = train_label.mean(dim=1, keepdim=True).to(device)

    # test_label = torchvision.io.read_image('../../data4/chuangzhi1.png')
    test_label = torchvision.io.read_image('../../data/Earthworm_20X_S1_C2_CV_0_new.png')
    c, h,w = test_label.shape[-3:]
    test_label = test_label.reshape(1,c,h,w).type(torch.float32) / 255

    test_inp = test_label.mean(dim=1, keepdim=True).to(device)
    return train_label, train_inp, test_label, test_inp

_log_path = None
def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


save_folder = os.path.join(save_base, tag)

os.makedirs(save_folder, exist_ok=True)
os.system(f'cp -r ./*.py {save_folder}')
set_log_path(save_folder)
log('current workdir is ' + os.getcwd())
log(f'back up successfully in {save_folder}')
command = 'python ' + ' '.join(sys.argv)
log('running command: ')
log(command)

def pad_to(x, stride=[1,1]):
    # https://www.coder.work/article/7536422
    # just match size to 2^n

    h, w = x.shape[-2:]
    

    if h % stride[-2] > 0:
        new_h = h + stride[-2] - h % stride[-2]
    else:
        new_h = h
    if w % stride[-1] > 0:
        new_w = w + stride[-1] - w % stride[-1]
    else:
        new_w = w

    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    assert (len(pad) + 1) % 2, 'pad length should be an even number'
    x = x[:,..., pad[2]:x.shape[-2]-pad[3], pad[0]:x.shape[-1]-pad[1]  ]

    return x


class UNET(nn.Module):
    def __init__(self, channels, kSize=3,ndims=3, negative_slope=0.1, usingbias=True):
        super().__init__()
        maxpoolblock = getattr(nn, 'MaxPool%dd' %ndims)
        interpolationMode = 'bilinear'  if ndims == 2 else 'trilinear'
        self.pooling = maxpoolblock(2)
        self.interpolation = lambda x:F.interpolate(x, scale_factor=2, mode=interpolationMode, align_corners=False)

        conv = getattr(nn, 'Conv%dd' %ndims)

        self.down_layers = nn.Sequential(*[
            ConvBlock(channels[ii], channels[ii+1], ndims = ndims, kSize=kSize, negative_slope=negative_slope, usingbias=usingbias)
            for ii in range(0, len(channels)-1, 1)
        ])
        
        self.up_layers = [ConvBlock(channels[-1], channels[-1], ndims = ndims, kSize=kSize, negative_slope=negative_slope, usingbias=usingbias)]
        for ii in range(len(channels)-1, 0, -1):
            self.up_layers.append(ConvBlock(channels[ii]*2, channels[ii-1], ndims = ndims, kSize=kSize, negative_slope=negative_slope, usingbias=usingbias))

        self.up_layers = nn.Sequential(*self.up_layers)
    def forward(self,x):
        x, pads = pad_to(x, [ 2**len(self.down_layers), 2**len(self.down_layers)] )
        x_history = []
        for level, down_layer in enumerate( self.down_layers):
            x = down_layer(x)
            x_history.append(x)
            x = self.pooling(x)

        for level, up_layer in enumerate( self.up_layers):
            x = up_layer(x)
            if len(x_history):
                x = torch.cat( (self.interpolation(x), x_history.pop()),dim=1 )

        x = unpad(x,pads)
        return x

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, ndims = 3, kSize=3, negative_slope=0.1, usingbias=False):
        super().__init__()
        convblock = getattr(nn, 'Conv%dd' %ndims)
        self.conv = nn.Sequential(*[
            convblock(inChannels, outChannels, kSize, padding=(kSize-1)//2, stride=1, bias=usingbias),
            # nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(negative_slope, inplace=True)
        ])
    def forward(self,x):
        y = self.conv(x)
        return y

def make_coord(shape, ranges=None, flatten=True,device='cpu'):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device, dtype=torch.float32)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class ColorNet(nn.Module):
    """
    A conv network.
    """
    def __init__(self, using_kernel_size=3, usingbias=True):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=using_kernel_size, stride=1, padding=using_kernel_size//2, bias=usingbias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            UNET(channels=[8,8,16, 16, 32], kSize=3, ndims=2, negative_slope=0.1, usingbias=usingbias),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=using_kernel_size, stride=1, padding=using_kernel_size//2, bias=usingbias),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='202505/')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    train_label, train_inp, test_label, test_inp = load_data()
    model = ColorNet(using_kernel_size=using_kernel_size, usingbias=True).to(device)
    from tqdm import tqdm

    loss_epoch = torch.zeros((epochs, ))
    train_accuracy_epoch = torch.zeros((epochs, 1))

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate )
    lr_scheduler = None
    # lr_scheduler = MultiStepLR(optimizer, milestones=multi_step_lr, gamma=0.5)

    loss_fn = torch.nn.L1Loss().to(device)
    epoch = 0
    if (epoch) % 50 == 0:
        model.eval()
        with torch.no_grad():
            train_pred = model(train_inp)
            os.makedirs(os.path.join(save_folder, 'train'), exist_ok=True)
            train_pred_tosave = (train_pred.detach().squeeze(0)*255).clamp(0,255).type(torch.uint8).cpu()
            torchvision.io.write_png(train_pred_tosave, os.path.join(save_folder, 'train', 'iter%06d.png'%epoch) )
            test_pred = model(test_inp)
            test_pred_tosave = (test_pred.detach().squeeze(0)*255).clamp(0,255).type(torch.uint8).cpu()
            os.makedirs(os.path.join(save_folder, 'test'), exist_ok=True)
            torchvision.io.write_png(test_pred_tosave, os.path.join(save_folder, 'test', 'iter%06d.png'%epoch) )

    for epoch in tqdm(range(0, epochs, 1), leave=False, desc='train', ncols=0):
        model.train()
        optimizer.zero_grad()

        train_pred = model(train_inp)
        loss = loss_fn( train_pred[:,:,:,:], train_label[:,:,:,:]) 
        
        loss_epoch[epoch] = loss.item()
        loss.backward()
        optimizer.step()
        model.eval()
        if (epoch+1) % epoch_check == 0:
            os.makedirs(os.path.join(save_folder, 'train'), exist_ok=True)
            train_pred_tosave = (train_pred.detach().squeeze(0)*255).clamp(0,255).type(torch.uint8).cpu()
            torchvision.io.write_png(train_pred_tosave, os.path.join(save_folder, 'train', 'iter%06d.png'%(epoch+1)) )
            with torch.no_grad():
                test_pred = model(test_inp)
                test_pred_tosave = (test_pred.detach().squeeze(0)*255).clamp(0,255).type(torch.uint8).cpu()
                os.makedirs(os.path.join(save_folder, 'test'), exist_ok=True)
                torchvision.io.write_png(test_pred_tosave, os.path.join(save_folder, 'test', 'iter%06d.png'%(epoch+1)) )

        if lr_scheduler is not None:
                lr_scheduler.step()


    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}, os.path.join(save_folder,'epoch-%d.pth'%epoch))