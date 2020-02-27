'''
This file implements JPP-Net for human parsing and pose detection.
'''

import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

import time

class JPP(object):
    
    # Magic numbers are for normalization. You can get details from original JPP-Net repo.
    IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
    def __init__(self, pb_path):
        options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=options)) 
        self.sess = tf.Session()
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='') # import compute graph
        self.sess.run(tf.global_variables_initializer())
        self.img_tensor = sess.graph.get_tensor_by_name('img_1:0')
        self.pose_tensor = sess.graph.get_tensor_by_name('pose:0')
        self.parse_tensor = sess.graph.get_tensor_by_name('parse:0')
        
        
    def predict(self, img):
        '''
        img is a human image array with shape (any,any,3)
        return a list, [pose, parse]
        '''
        ret = self.sess.run([self.pose_tensor,self.parse_tensor],  feed_dict={self.img_tensor: img-JPP.IMG_MEAN})
        return ret