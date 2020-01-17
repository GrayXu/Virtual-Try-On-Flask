'''
This file implements the results from server to clients' requests
'''
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

parser = optparse.OptionParser()

parser.add_option(
    '-u', '--url', default='http://localhost:5000/cloth', dest='url')


def test():
    # Use base64 to encode & decode image to transfer on the Internet
    options, args = parser.parse_args()
    img = cv2.imread('./example/cloth.jpg')
    img_str = cv2.imencode('.jpg', img)[1]
    imagebase64_c = base64.b64encode(img_str)
    img = cv2.imread('./example/human.jpg')
    img_str = cv2.imencode('.jpg', img)[1]
    imagebase64_h = base64.b64encode(img_str)

    data_str = {
        "uuid": "1.jpg",
        "image_person": imagebase64_h,
        "image_cloth": imagebase64_c,
        "format": "jpg",
    }
    num = 0
    res = requests.post(options.url, data=json.dumps(data_str))
    print(res.status_code)
    num += 1
    print(res.content)


if __name__ == '__main__':
    threads = []
    for x in range(1):
        t = threading.Thread(target=test)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
