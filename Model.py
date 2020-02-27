'''
本类为虚拟试衣的整合，提供两个接口函数，初始化后，即可进行前向预测。
This file implements the whole virtual try-on networks.
After initiation with init(pathes...), you can call predict(...)
'''
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from CPVTON import CPVTON
from JPPNet import JPP
import torch
import time
from PIL import ImageDraw, ImageEnhance
import torchvision.transforms as transforms
import cv2


class Model(object):

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __init__(self, pb_path, gmm_path, tom_path, use_cuda=True):
        '''
        传入三个参数，分别是JPP、GMM、TOM三个模型的预训练文件路径.
        parameters: 3 pre-trained model(JPP, GMM, TOM) files' pathes
        '''
        self.jpp = JPP(pb_path)
        self.cpvton = CPVTON(gmm_path, tom_path, use_cuda=use_cuda)

    def predict(self, human_img, c_img, need_pre=True, need_bright=False, keep_back=False, need_dilate=False, check_dirty=False):
        '''
        输入：
            human_img为人体图片，c_img为衣服图片,均为numpy array with shape(256,192,3) RGB
            五个flag:
                need_pre为是否需要预处理（crop+resize到256*192），在LIP数据集上关闭need_pre效果较好（upsample损失）
                need bright为亮度增强，
                keep back为基于mask保持除了上衣以外的部分，
                need dilate为膨胀柔化keep back的mask，需要keep_back开启, 
                check_dirty为增加检查数据是否有肢体遮挡交叉（可能检测不全）  

        返回：
            一个(256,192,3)的穿上衣服的图片，
            画面中人体的置信度，0则为触发强排除条件

        parameters: 
            human_img: human's image
            c_img: cloth image with the shape of (256,192,3) RGB
            need_pre: need preprocessing, including crop and zoom 
            need_bright: brightness enhancement
            keep_back: keep image's background
            need_dilate: if keep_back is True, and you need dilate the mask
            check_dirty: check limb cross
        return: 
            a (256, 192, 3) image with specified people wearing specified clothes,
            confidence of this image
        '''
        if need_bright:
            enh_bri = ImageEnhance.Brightness(Image.fromarray(human_img))
            human_img = np.array(enh_bri.enhance(1.3))

        result = self.jpp.predict(human_img)
        pose = result[0][0]
        parse = result[1][0]

        pose_data, trusts = self.__getPoseData__(pose)

        if need_pre:
            human_img, pose_data, parse = self.__cropByPoseData__(
                human_img, pose_data, parse)

        if pose_data is None:  # no person and pose data error
            return human_img, 0.0
        # drity data(e.g. cross hand)?
        if check_dirty and self.__is_dirty__(pose_data[10], pose_data[15], pose_data[12], pose_data[2], pose_data[3], pose_data[13]):
            return human_img, 0.0

        pose_map = self.__getPoseMap__(pose_data)

        # treat left & right legs as trousers
        parse = parse + np.array(parse == 16, dtype='uint8') * \
            (-(16-9))+np.array(parse == 17, dtype='uint8')*(-(17-9))
        parse = np.array(parse[:, :, 0], dtype="uint8")
        (out, warp) = self.cpvton.predict(parse, pose_map, human_img, c_img)

        out_img = np.array((np.transpose(out.detach().cpu().numpy()[
                           0], axes=(1, 2, 0))+1)/2*255, dtype='uint8')

        if keep_back:
            # 保留手臂和背景
            # keep arms & background
            if len(parse.shape) == 2:
                parse = parse.reshape((256, 192, 1))
            cloth_mask = np.array(parse == 5, dtype='float32')

            if need_dilate:  # revise mask
                # 膨胀后边缘虚化
                # Edge emptiness after dilate
                cloth = cloth_mask[:, :, 0]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
                dilated = cv2.dilate(cloth, kernel)
                dilated = cv2.blur(dilated, (14, 14))

                new_cloth = Image.fromarray((dilated*255))
                new_cloth = new_cloth.resize(
                    (192//10, 256//10), Image.BILINEAR)
                new_cloth = new_cloth.resize((192, 256), Image.BILINEAR)

                new_cm = np.array(new_cloth)
                new_cm = np.array(new_cm/255, dtype='float32')
                cloth_mask = np.resize(new_cm, (256, 192, 1))

            out_img = human_img*(1-cloth_mask)+out_img*cloth_mask

        return np.array(out_img, dtype='uint8'), trusts

    def __getPoseData__(self, pose):
        '''
        输入:jppnet输出的pose结果
        输出：
            CPVTON所需要的pose
            和根据pose估算得到的置信度
        置信度计算规则为：检测上半身关键点共12个点八邻域均值的min值, threshold大概在0.1

        parameters: JPP-Net's output, pose keypoints
        return: Pose data, a part of CP-VTON's input.
                And result's confidence, computed by the min value of 12 eight-neighborhood keypoints' confidence 
                (its threshold is about 0.1)
        '''
        contents = []
        trusts = []
        for i in range(16):
            tmp = np.argmax(pose[:, :, i])
            y = tmp % pose.shape[1]
            x = tmp//pose.shape[1]
            contents.append([y, x, 1])
            if i not in [0, 1, 4, 5]:
                trusts.append(sum([pose[x, y, i],
                                   pose[x+1, y, i],
                                   pose[x, y+1, i],
                                   pose[x-1, y, i],
                                   pose[x, y-1, i],
                                   pose[x+1, y-1, i],
                                   pose[x-1, y+1, i],
                                   pose[x+1, y+1, i],
                                   pose[x-1, y+1, i],
                                   ])/8)

        return np.array(contents), min(trusts)

    def __getPoseMap__(self, pose_data):
        '''
        传入的pose_data也为np array, shape为(16,3)
        返回：对所有的pose位置，绘制一个大小为3*3的白色正方形，作为人体特征的一部分

        parameters: pose data with the shape of (16,3) [x,y,confidence]
        return: a pose map array with the shape (?, ?, 3) from drawing a 3*3 white (max value) square at every pose position as a part of human feature
        '''
        im_pose = Image.new('L', (192, 256))
        pose_draw = ImageDraw.Draw(im_pose)
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, 256, 192)
        for i in range(point_num):
            one_map = Image.new('L', (192, 256))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            r = 3
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx +
                                r, pointy+r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = Model.transformer(one_map)
            pose_map[i] = one_map[0]

        return pose_map

    def __cropByPoseData__(self, img, pose_data, parse):
        '''
        根据pose的位置进行裁剪缩放出256*192的图片
        规则为，根据最高点和最低点pose进行在新图片height比例0.2大小的上下扩展，裁剪后进行缩放‬
        返回：
            crop&resize后的新图片，
            更新后的pose data，
            更新后的parse

        crop and scale picture to get 256*192 resolution
        rules: based on the hightest & lowest pose, crop 120% distance and zoom it to right scale
        return: picture after operation,
                pose data after operation,
                parse results after operation
        '''
        h, w = img.shape[0], img.shape[1]
        height = max([pose_data[2][1], pose_data[3][1],
                      pose_data[10][1], pose_data[15][1]]) - pose_data[9][1]

        # 上下的裁剪位置
        # up & low position
        pre_height = max([pose_data[2][1], pose_data[3][1],
                          pose_data[10][1], pose_data[15][1]])-pose_data[9][1]
        upper = max(pose_data[9][1]-int(pre_height*0.2), 0)
        bounder = min(max([pose_data[2][1], pose_data[3][1],
                           pose_data[10][1], pose_data[15][1]])+int(pre_height*0.2), h)

        # 左右的裁剪位置
        # left & right position
        height = bounder - upper
        width = int(height/4*3)

        left = min([pose_data[12][0], pose_data[11][0], pose_data[10][0]])
        right = max([pose_data[13][0], pose_data[14][0], pose_data[15][0]])
        change = (width - (right-left))/2

        if left >= 0+change and right <= w-change:
            left -= change
            right += change
        elif left < 0+change:
            left = 0
            right = min(right+change+change-left, w)
        elif right > w-change:
            right = w-1
            left = max(left-change-(change-(w-right)), 0)
        else:
            # 裁剪不了 can't crop
            return None

        left = int(left)
        right = int(right)
        print("upper:%d,bounder:%d,left:%d,right:%d" %
              (upper, bounder, left, right))

        if left >= right or upper >= bounder:
            print("no person")
            return img, None, None

        factor_h = h/height
        factor_w = w/width

        for i in range(16):
            pose_data[i][0] = int((left+pose_data[i][0])*factor_w)
            pose_data[i][1] = int((upper+pose_data[i][1])*factor_h)

        new_img = np.array(img[upper:bounder, left:right, :])

        # 裁剪parse结果。注意parse的结果会有比较大损失
        # crop parse result. WARNING! An apparent loss would happen here
        parse = parse[upper:bounder, left:right, :]

        parse = np.array(Image.fromarray(np.array(np.concatenate(
            [parse, parse, parse], axis=2)/17*255, dtype='uint8')).resize((192, 256)))

        parse = np.array(parse[:, :, :1]/(255/17), dtype='uint8')

        print("after crop shape:"+str(new_img.shape))
        return np.array(Image.fromarray(new_img).resize((192, 256))), pose_data, parse

    def __get_K_b__(self, b, c):
        '''
        计算直线的斜率K与偏移bias
        get gradient K and bias b of the line
        '''
        if b[0] == c[0]:
            K = 99999999
        else:
            K = (b[1]-c[1])/(b[0]-c[0])
        B = b[1]-K*b[0]
        return (K, B)

    def __upon_line__(self, a, KB):
        '''
        a点是否高于特定直线 with K,b
        point A is higher than the line with specified gradient K and bias b?
        '''
        K, B = KB
        if K*a[0]+B > a[1]:
            return True
        else:
            return False

    # reverse
    def __right_line__(self, a, KB, x):
        '''
        a点是否在特定直线（KB）的右侧。
        x为直线上任意一点的横坐标，用于在直线垂直的case上获得左右关系

        point A is on the left side of the line with specified gradient K and bias b?
        x is a random point's value on the abscissa, which would be used for vertical line case
        '''
        K, B = KB

        if K == 99999999:
            if a[0] > x+5:
                return True
            else:
                return False

        result = K*a[0]+B >= a[1]
        if (result > a[1] and K > 0) or (result < a[1] and K < 0):
            return True
        else:
            return False

    def __is_dirty__(self, wrist_a, wrist_b, a, b, c, d):
        '''
        judge this img is dirty or not
        two points, wrist_a and wrist_b, are in the rectangle a-b-c-d?
        a->leftup, b->leftdown, c->rightdown, d->rightup
        Note: For JPP Net's output poses, the order is [10,15,12,2,3,13]
        '''
        # 手腕部接近四个点则不算dirty
        # not dirty if wrists are close to these four points
        margin = 250
        KB_list = [self.__get_K_b__(a, b), self.__get_K_b__(
            b, c), self.__get_K_b__(c, d), self.__get_K_b__(a, d)]

        if np.sum((np.sum(np.array(wrist_a)-np.array(b)))**2) <= margin or np.sum((np.sum(np.array(wrist_b)-np.array(c)))**2) <= margin:
            return False  # too close to standard points
        if self.__right_line__(wrist_a, KB_list[0], wrist_a[0]) and self.__upon_line__(wrist_a, KB_list[1]):
            print("wrist_a dirty")
            return True
        if not self.__right_line__(wrist_b, KB_list[2], wrist_b[0]) and self.__upon_line__(wrist_b, KB_list[1]):
            print("wrist_b dirty")
            return True
        else:
            return False
