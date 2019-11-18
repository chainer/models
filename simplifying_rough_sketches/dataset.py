from PIL import Image
import numpy as np

from chainer.dataset import dataset_mixin

class CustomDataset(dataset_mixin.DatasetMixin):

    def __init__(self, input_path, target_path, height=424, width=424):
        self.input_img = input_path
        self.target_img = target_path
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.input_img)
    
    def preprocess(self, img_path):
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize((self.height, self.width))
        img = np.asarray(img)/255.0
        img = img.reshape(1,self.height, self.width)
        return img

    def __getitem__(self, index):
        X = self.preprocess(self.input_img[index])
        y = self.preprocess(self.target_img[index])

        return X, y
    
    def get_example(self, index):
        X = self.preprocess(self.input_img[index])
        y = self.preprocess(self.target_img[index])

        return X, y
