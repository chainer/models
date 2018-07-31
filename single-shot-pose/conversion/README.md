# Conversion from PyTorch weights to Chainer

|object| converted weight| converted initial weight |
|--:|--:|------:|
|ape|   (a) [link](https://drive.google.com/file/d/1ry8uJW5JvNK_j8yYvwtz9ebLPxYdSqtd/view?usp=sharing)     |  (b)  [link](https://drive.google.com/file/d/17wKAL5NcTtPExtwp4kOU--Ul7-wrPSE6/view?usp=sharing)   |
|benchvise | (c) [link](https://drive.google.com/file/d/1QZ2OKHPrc2VWt3bi257uQqa155TxImpo/view?usp=sharing) | (d) [link](https://drive.google.com/file/d/19P3U4Kki3byn_fTZ_vJnT702D0hdD4EX/view?usp=sharing) |

ImageNet pretrained YOLOv2: (e) [link](https://drive.google.com/file/d/1rLnKBQviEIIeqp8VD-LVi0LEeXGsO3TN/view?usp=sharing)

### PyTorch weights
```bash
wget -O backup.tar --no-check-certificate "https://onedrive.live.com/download?cid=0C78B7DE6C569D7B&resid=C78B7DE6C569D7B%21191&authkey=AP183o4PlczZR78"

# ImageNet pretrained DarkNet
wget https://pjreddie.com/media/files/darknet19_448.conv.23 .
```

### Conversion

```bash
$ python torch_to_npz.py [--cfg cfgfile] [--weight weightfile] [--skip-last]

```

The pretrained weights can be converted from pytorch models by the following commands.

```bash
(a) python torch_to_npz.py --weight backup/ape/model_backup.weights --output ssp_yolo_v2_linemod_ape_converted.npz
(b) python torch_to_npz.py --weight backup/ape/init.weights --skip-last --output ssp_yolo_v2_linemod_ape_init_converted.npz
(c) python torch_to_npz.py --weight backup/benchvise/model_backup.weights --output ssp_yolo_v2_linemod_benchvise_converted.npz
(d) python torch_to_npz.py --weight backup/benchvise/init.weights --skip-last --output ssp_yolo_v2_linemod_benchvise_init_converted.npz
(e) python torch_to_npz.py --cfg cfg/yolo-pose-pre.cfg --weight darknet19_448.conv.23 --skip-last --output ssp_yolo_v2_imagenet_converted.npz
```
