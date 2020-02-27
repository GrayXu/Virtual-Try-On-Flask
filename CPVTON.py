# coding=utf-8
'''
CPVTON的模型定义接口。

This file implements CP-VTON.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import argparse
import os
import time
import sys
from networks import GMM, UnetGenerator, load_checkpoint
import json

# 为输入提供正则化
# normalize inputs
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CPVTON(object):

    def __init__(self, gmm_path, tom_path, use_cuda=True):
        '''
        初始化两个模型的预训练数据
        init pretrained models
        '''
        self.use_cuda = use_cuda
        self.gmm = GMM(use_cuda=use_cuda)
        load_checkpoint(self.gmm, gmm_path, use_cuda=use_cuda)
        self.gmm.eval()
        self.tom = UnetGenerator(
            23, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(self.tom, tom_path, use_cuda=use_cuda)
        self.tom.eval()
        if use_cuda:
            self.gmm.cuda()
            self.tom.cuda()
        print("use_cuda = "+str(self.use_cuda))

    def predict(self, parse_array, pose_map, human, c):
        '''
        传入的前四个都是array. shape为(*,256,192)
        input 4 np array with the shape of (*,256,192)
        '''
        im = transformer(human)
        c = transformer(c)  # [-1,1]

        # parse -> shape

        parse_shape = (parse_array > 0).astype(np.float32)

        # 模糊化，下采样+上采样
        # blur, downsample + upsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((192//16, 256//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((192, 256), Image.BILINEAR)
        shape = transformer(parse_shape)

        parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 4).astype(np.float32) + \
            (parse_array == 13).astype(np.float32) + \
            (parse_array == 9).astype(np.float32)
        phead = torch.from_numpy(parse_head)  # [0,1]
        im_h = im * phead - (1 - phead)

        agnostic = torch.cat([shape, im_h, pose_map], 0)

        if self.use_cuda:
            # batch==1
            agnostic = agnostic.unsqueeze(0).cuda()
            c = c.unsqueeze(0).cuda()
            # warp result
            grid, theta = self.gmm(agnostic.cuda(), c.cuda())
            c_warp = F.grid_sample(c.cuda(), grid, padding_mode='border')
        else:
            agnostic = agnostic.unsqueeze(0)
            c = c.unsqueeze(0)
            grid, theta = self.gmm(agnostic, c)
            c_warp = F.grid_sample(c, grid, padding_mode='border')

        tensor = (c_warp.detach().clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)
        array = tensor.numpy().astype('uint8')

        c_warp = transformer(np.transpose(array[0], axes=(1, 2, 0)))
        c_warp = c_warp.unsqueeze(0)

        if self.use_cuda:
            outputs = self.tom(torch.cat([agnostic.cuda(), c_warp.cuda()], 1))
        else:
            outputs = self.tom(torch.cat([agnostic, c_warp], 1))

        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        if self.use_cuda:
            p_tryon = c_warp.cuda() * m_composite + p_rendered * (1 - m_composite)
        else:
            p_tryon = c_warp * m_composite + p_rendered * (1 - m_composite)
            

        return (p_tryon, c_warp)
