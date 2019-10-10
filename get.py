#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-7-11 下午4:43
# @Author  : zhangzhen201
# @Site    :
# @File    : HttpClient.py
# @Software: PyCharm Edu


import requests
import optparse
import cv2
import numpy as np
import sys
import threading
import os
from os.path import basename
import json
import base64
import time

parser = optparse.OptionParser()

parser.add_option('-u', '--url', default='http://localhost:5000/cloth', dest = 'url')

def test():
    options, args = parser.parse_args()
    img = cv2.imread('./example/cloth.jpg')
    img_str = cv2.imencode('.jpg', img)[1]
    imagebase64_c = base64.b64encode(img_str)
    img = cv2.imread('./example/human.jpg')
    img_str = cv2.imencode('.jpg', img)[1]
    imagebase64_h = base64.b64encode(img_str)
    # with open("./vto/example/cloth.jpg") as f:
        # x = f.read()
        # imagebase64_c = base64.b64encode(x)
    # with open("./vto/example/human.jpg") as f:
        # x = f.read()
        # imagebase64_h = base64.b64encode(x)
    data_str = {
        "uuid": "1.jpg",
        "image_person": imagebase64_h,
        "image_cloth": imagebase64_c,
        "format": "jpg",
    }
    num = 0
    while True:
        t0 = time.time()
        res = requests.post(options.url, data = json.dumps(data_str))
        t1 = time.time()
        print t1 - t0
        print(res.status_code)
        num += 1
        print(res.content)
        break
        if num > 10: break


if __name__ == '__main__':
    # test()
    threads=[]
    for x in range(1):
        t=threading.Thread(target=test)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
