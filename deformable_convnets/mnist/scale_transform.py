from chainercv import transforms


def transform(in_data):
    img, label = in_data
    img = transforms.random_expand(img, max_ratio=3)
    img = transforms.resize(img, (28, 28))
    return img, label
