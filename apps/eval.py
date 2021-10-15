import sys
import os

from numpy.lib.function_base import select

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC
    
    def load_image(self, images, masks):
        # Name
        img_name = os.path.splitext(os.path.basename(images[0]))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        
        # modify: multi-view setting
        calibList = []
        if self.opt.num_views == 1:
            calibList.append(torch.Tensor(projection_matrix).float())
        elif self.opt.num_views == 3:
            extrin_60 = np.array([
                [-0.5, 0, 0.866, 0], 
                [0, 1, 0, 0],
                [-0.866, 0, -0.5, 0], 
                [0, 0, 0, 1]
            ])
            extrin_120 = np.array([
                [-0.5, 0, -0.866, 0], 
                [0, 1, 0, 0],
                [0.866, 0, -0.5, 0], 
                [0, 0, 0, 1]
            ])
            calibList.append(torch.Tensor(projection_matrix).float())
            calibList.append(torch.Tensor(np.matmul(projection_matrix, extrin_60)).float())
            calibList.append(torch.Tensor(np.matmul(projection_matrix, extrin_120)).float())
        elif self.opt.num_views == 4:
            extrin_90 = np.array([
                [0, 0, 1, 0], 
                [0, 1, 0, 0],
                [-1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            extrin_180 = np.array([
                [-1, 0, 0, 0], 
                [0, 1, 0, 0],
                [0, 0, -1, 0], 
                [0, 0, 0, 1]
            ])
            extrin_270 = np.array([
                [0, 0, -1, 0], 
                [0, 1, 0, 0],
                [1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            calibList.append(torch.Tensor(projection_matrix).float())
            calibList.append(torch.Tensor(np.matmul(projection_matrix, extrin_90)).float())
            calibList.append(torch.Tensor(np.matmul(projection_matrix, extrin_180)).float())
            calibList.append(torch.Tensor(np.matmul(projection_matrix, extrin_270)).float())
        # Mask
        maskList = []
        imageList = []
        for mask, image in zip(masks, images):
            mask = Image.open(mask).convert('L')
            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            maskList.append(mask)
            image = Image.open(image).convert('RGB')
            image = self.to_tensor(image)
            image = mask.expand_as(image) * image
            imageList.append(image)
        return {
            'name': img_name,
            'img': torch.stack(imageList, dim=0),
            'calib': torch.stack(calibList, dim=0),
            'mask': torch.stack(maskList, dim=0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }
        #--------------------------------------#

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    #test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    #test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    
    # modify: multi-view setting #
    test_images = [opt.test_folder_path+'/0_0_00.jpg', opt.test_folder_path+'/90_0_00.jpg', opt.test_folder_path+'/180_0_00.jpg', opt.test_folder_path+'/270_0_00.jpg']
    test_masks = [opt.test_folder_path+'/0_0_00_mask.png', opt.test_folder_path+'/90_0_00_mask.png', opt.test_folder_path+'/180_0_00_mask.png', opt.test_folder_path+'/270_0_00_mask.png']
    #--------------------------#

    #test_images = [opt.test_folder_path+'/0_0_00.jpg']
    #test_masks = [opt.test_folder_path+'/0_0_00.png']
    print("Use view:", opt.num_views)

    #for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
    try:
        data = evaluator.load_image(test_images, test_masks)
        evaluator.eval(data, True)
    except Exception as e:
        print("error:", e.args)
