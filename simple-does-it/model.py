from chainercv.links import ResNet50
from chainercv.links import Conv2DBNActiv
from chainercv.experimental.links.model.pspnet import PSPNet


def get_dilated_resnet50():

    imagenet_resnet = ResNet50(pretrained_model='imagenet', arch='he')
    imagenet_resnet.res4.a.conv1.conv.stride = (1, 1)
    imagenet_resnet.res4.a.residual_conv.conv.stride = (1, 1)
    imagenet_resnet.res5.a.conv1.conv.stride = (1, 1)
    imagenet_resnet.res5.a.residual_conv.conv.stride = (1, 1)

    def get_dilated_cbr(prev):
        new = Conv2DBNActiv(prev.conv.W.shape[1], prev.conv.W.shape[0], 3,
                            1, 2, 2, nobias=True)
        new.conv.W.data[:] = prev.conv.W.data[:]
        return new

    imagenet_resnet.res4.a.conv2 = get_dilated_cbr(imagenet_resnet.res4.a.conv2)
    imagenet_resnet.res4.b1.conv2 = get_dilated_cbr(imagenet_resnet.res4.b1.conv2)
    imagenet_resnet.res4.b2.conv2 = get_dilated_cbr(imagenet_resnet.res4.b2.conv2)
    imagenet_resnet.res4.b3.conv2 = get_dilated_cbr(imagenet_resnet.res4.b3.conv2)
    imagenet_resnet.res4.b4.conv2 = get_dilated_cbr(imagenet_resnet.res4.b4.conv2)
    imagenet_resnet.res4.b5.conv2 = get_dilated_cbr(imagenet_resnet.res4.b5.conv2)

    imagenet_resnet.res5.a.conv2 = get_dilated_cbr(imagenet_resnet.res5.a.conv2)
    imagenet_resnet.res5.b1.conv2 = get_dilated_cbr(imagenet_resnet.res5.b1.conv2)
    imagenet_resnet.res5.b2.conv2 = get_dilated_cbr(imagenet_resnet.res5.b2.conv2)

    imagenet_resnet.pick = ['res4', 'res5']
    imagenet_resnet.remove_unused()
    return imagenet_resnet


def get_pspnet_resnet50(n_class):
    return PSPNet(get_dilated_resnet50(), n_class, (448, 448), bn_kwargs={})
