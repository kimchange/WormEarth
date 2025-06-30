import torch.nn as nn
import torch

import os,sys
import numpy as np
import torchvision
import torch.nn.functional as F
using_kernel_size = 3
device = 'cpu'
for jj in range(1,2):
    # tag = 'torch_20250531_unet_relu_down32_lrdecay_sgd_train%d'%ii
    tag = 'torch_20250620_unet_relu_adam_wormearth20new'
    workdir = '../save/' + tag +'/'
    sys.path.append(workdir)
    from wormearth import ColorNet

    model = ColorNet(using_kernel_size=using_kernel_size, usingbias=True).to(device)

    checkpoint = torch.load(workdir+'epoch-999.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sourcefolder = '../data/'
    save_folder = './results/' + tag +'_data/'
    all_tests = os.listdir(sourcefolder)

    all_tests = ['Earthworm_20X_S1_C2_CV_0_new.png']
    test_label = torchvision.io.read_image(sourcefolder+all_tests[0]).unsqueeze(0).type(torch.float32).to(device) / 255
    # scale_list = -torch.cos(torch.arange(1000) /1000*2*torch.pi) * 0.3 + 0.7
    for ii in range(1):

        test_inp = test_label.mean(dim=1, keepdim=True)
        with torch.no_grad():
            test_pred = model(test_inp)
        # test_pred_tosave = F.interpolate(test_pred, size=test_label.shape[-2:], align_corners=False, mode='bicubic')
        test_pred_tosave = (test_pred.detach().squeeze(0)*255).clamp(0,255).type(torch.uint8).cpu()
        os.makedirs(save_folder, exist_ok=True)
        torchvision.io.write_png(test_pred_tosave, os.path.join(save_folder, all_tests[0][0:-4] + '_t%03d.png'%(ii)  )   )