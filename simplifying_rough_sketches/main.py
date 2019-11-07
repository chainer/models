import chainer

import time
import argparse
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import Net
from dataset import CustomDataset
from read_data import get_data
from utils import samples
from train_val import train, validate
from utils import argument_parser, create_directory
from predict import predict

def main():

    args = argument_parser()

    gpu_id = args.gpu_id
    # root = args.root
    height = args.height
    width = args.width
    BATCHSIZE = args.batch_size
    N_FOLD = args.n_folds
    num_epochs = args.num_epochs

    out = args.out
    create_directory(out)

    root = args.root
    Input = os.path.join(root, 'Input')
    Target = os.path.join(root,'Target')
    # height = width = 424

    input_images, target_images = get_data(Input, Target)


    if args.samples:
        show_samples(input_images)
        show_samples(target_images)


    if args.predict and args.load_model:

        model_path = args.load_model
        img_path = args.predict

        model = Net()

        from chainer import serializers
        serializers.load_npz(model_path, model)
        print ('Model Loaded')

        
        img = predict(model, img_path, height, width, gpu_id)
        pilimg = Image.fromarray(np.uint8(img))
        pilimg.save(os.path.join(out,f'predict{hash(img_path)}.png'))
        print (f'Image saved in {out} directory.')


    elif args.train:

        print ('Training Started.')

        model = Net()

        if gpu_id >= 0:
            chainer.backends.cuda.get_device_from_id(gpu_id).use()
            model.to_gpu()
            print ('Switched to CUDA backend')
        else:
            print ('Training on CPU')


        optimizer = chainer.optimizers.AdaDelta(rho=0.9)
        optimizer.setup(model)
        optimizer.use_cleargrads(True)

        dataset = CustomDataset(input_path=input_images, target_path=target_images, height=height, width=width)
        train_val_set = chainer.datasets.get_cross_validation_datasets_random(dataset=dataset, n_fold=N_FOLD, seed=None)

        total_loss_val, total_loss_train = [],[]

        import time
        since = time.time()
        epoch_num = num_epochs
        best_val_loss = 1000

        for fold, train_val in enumerate(train_val_set):

            train_set, val_set = train_val

            train_loader = chainer.iterators.MultithreadIterator(dataset=train_set, batch_size=BATCHSIZE, shuffle=True, repeat=False)
            val_loader = chainer.iterators.MultithreadIterator(dataset=val_set, batch_size=BATCHSIZE, shuffle=True, repeat=False)

            for epoch in range(1, epoch_num+1):
                # chainer.cuda.memory_pool.free_all_blocks()
                avg_loss_train, loss_train = train(train_loader, model, optimizer, epoch, gpu_id)
                loss_val = validate(val_loader, model, optimizer, epoch, gpu_id)
                total_loss_val.append(loss_val)
                total_loss_train.append(avg_loss_train)

                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    print('*****************************************************')
                    print(f'best record: [Fold {fold}] [epoch {epoch}], [val loss {loss_val:.5f}]')
                    print('*****************************************************')

                if epoch%1 == 0:
                    print ('Fold: {0} Epoch: {1}'.format(fold,epoch))
                    samples(input_images[0], target_images[0], model, out, height, width, gpu_id)
                    img = predict(model, input_images[0], height, width, gpu_id)
                    pilimg = Image.fromarray(np.uint8(img))
                    pilimg.save(os.path.join(out,str(fold)+'_'+str(epoch)+'.png'))

        end = time.time()

        print ('Training Completed')
        if args.save_model:
            print('Saving model in models directory')
            from chainer import serializers
            create_directory('models')
            serializers.save_npz(os.path.join('models','simplifying_rough_sketches.model'), model)

        print ('Time Taken: ',end-since)
        fig = plt.figure(num = 2)
        fig1 = fig.add_subplot()
        fig1.plot(total_loss_train, label = 'training loss')
        fig1.plot(total_loss_val, label = 'validation loss')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(out,'loss.png'))
        plt.close(fig)


        try:
            for i in range (5):
                import random
                k = random.randint(1,63)
                samples(input_images[k], target_images[k], model, out)
        except Exception as e:
            pass

    else:
        print ('Exiting')

if __name__ == '__main__':
    main()