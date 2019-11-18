from utils import AverageMeter, length

import chainer
from chainer import Variable
import chainer.functions as F


def train(train_loader, model, optimizer, epoch, gpu_id):
    train_loader.reset()

    train_loss = AverageMeter()
    
    curr_iter = (epoch - 1) * length(train_loader)
    loss_train = []

    
    for i, data in enumerate(train_loader):
        
        model.cleargrads()
        
        batch_img, targets = chainer.dataset.concat_examples(data)
    
        inputs = Variable(batch_img.astype('float32'))
        targets = Variable(targets.astype('float32'))
    
        if gpu_id >= 0:
            chainer.backends.cuda.get_device_from_id(gpu_id).use()
            inputs.to_gpu()
            targets.to_gpu()
    
        outputs = model(inputs)
        loss = F.mean_squared_error(outputs, targets)
        train_loss.update(loss.item())
        loss.backward()
        optimizer.update()
        curr_iter += 1
        if epoch % 1 == 0:
            if (i + 1) % 8 == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                    epoch, i + 1, length(train_loader), train_loss.avg))
        loss_train.append(train_loss.avg)
    
    return train_loss.avg, loss_train


def validate(val_loader, model, optimizer, epoch, gpu_id):
    val_loss = AverageMeter()
    
    with chainer.no_backprop_mode():
        val_loader.reset()
        for _, data in enumerate(val_loader):
            images, targets = chainer.dataset.concat_examples(data)

            images = Variable(images.astype('float32'))            
            targets = Variable(targets.astype('float32'))
        
            if gpu_id >= 0:
                chainer.backends.cuda.get_device_from_id(gpu_id).use()
                images.to_gpu()
                targets.to_gpu()
            
            outputs = model(images)
            loss = F.mean_squared_error(outputs, targets)
            val_loss.update(loss.item())

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f]' % (epoch, val_loss.avg))
    print('------------------------------------------------------------')
    return val_loss.avg

