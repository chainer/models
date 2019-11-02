import chainer
from chainer import Variable

import numpy as np
import cv2

from utils import preprocess

def predict(model, img_path, height, width, gpu_id):
    with chainer.no_backprop_mode():
        in_shape = np.asarray(cv2.imread(img_path)).shape
        img = preprocess(img_path, height, width)
        fin_shape = np.asarray(img).shape
        img = img.reshape(1, 1, fin_shape[0],fin_shape[1])

        img = Variable(img.astype('float32'))

        if gpu_id >= 0:
            chainer.backends.cuda.get_device_from_id(gpu_id).use()
            img.to_gpu()

        output = model(img)
        output.to_cpu()
        img = np.array(output.data) * 255
        # img = img.astype('uint8')
        img = img.squeeze().squeeze()
        # print (img.shape)

        img = cv2.resize(img, (in_shape[1],in_shape[0]),interpolation = cv2.INTER_AREA)
        return img
