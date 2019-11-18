from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import math
import argparse
import os
import time

def length(train_loader): return math.ceil(len(train_loader.dataset)/train_loader.batch_size)


# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def preprocess(img_path, height, width):
    img = Image.open(img_path)
    img = img.convert('L')
    img = img.resize((height, width))
    img = np.asarray(img)
    return img


def samples(input_image, target_image, model, out, height, width, gpu_id):
    from predict import predict

    _,figs = plt.subplots(1,3)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.rcParams['figure.figsize'] = [15, 15]
    figs[0].set_title('Input')
    figs[0].imshow(Image.open(input_image),cmap = 'gray')

    figs[1].set_title('Prediction')
    figs[1].imshow(predict(model, input_image, height, width, gpu_id),cmap='gray')

    figs[2].set_title('Target')
    figs[2].imshow(Image.open(target_image),cmap = 'gray')

    plt.savefig(os.path.join(out,f'sample{time.time()}.png'))



def argument_parser():
    parser = argparse.ArgumentParser(description='Rough Sketch Simplification')
    parser.add_argument('--gpu_id', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', type=str, default='result', help='Directory to output the result')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch Size')
    parser.add_argument('--height', '-ht', type=int, default=64, help='height of the image to resize to')
    parser.add_argument('--width', '-wd', type=int, default=64, help='width of the image to resize to')
    parser.add_argument('--samples', type=bool, default=False, help='See sample training images')
    parser.add_argument('--num_epochs', type=int, default=75, help='Number of epochs to train on')
    parser.add_argument('--train', default=True, type=bool, help='train the model')
    parser.add_argument('--root', default='.', type=str, help='Root Directory for Input and Target images.')
    parser.add_argument('--n_folds', default=7, type=int, help='Number of folds in k-fold cross validation.')
    parser.add_argument('--save_model', default=True, type=bool, help='Save model after training.')
    parser.add_argument('--load_model', default=None, type=str, help='Load existing model.')
    parser.add_argument('--predict', default=None, type=str, help='Path of rough sketch to simplify using existing model')

    args = parser.parse_args()

    return args

def create_directory(path):
    try:
    # Create target Directory
        os.mkdir(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        print("Directory " , path ,  " already exists")
