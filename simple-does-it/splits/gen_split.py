import os
from chainercv.datasets.sbd import sbd_utils


data_dir = sbd_utils.get_sbd()

# https://raw.githubusercontent.com/martinkersner/train-DeepLab/master/exper/voc12/list/original/train_aug.txt
f = open('train_aug.txt')
ids = []
for l in f.readlines():
    path = l.split()[0]
    ids.append(os.path.split(os.path.splitext(path)[0])[1])

sbd_ids = []
voc_ids = []
for id_ in ids:
    if os.path.exists(os.path.join(data_dir, 'img', id_ + '.jpg')):
        sbd_ids.append(id_)
    else:
        voc_ids.append(id_)

with open('sbd_trainaug.txt', 'w') as f:
    for id_ in sbd_ids:
        f.writelines(id_ + '\n')
        
with open('voc_trainaug.txt', 'w') as f:
    for id_ in voc_ids:
        f.writelines(id_ + '\n')
