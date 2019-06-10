# Chainer Implementation of SiamRPN Family

The implementation is based on https://github.com/STVIR/pysot

## Dependencies

- Chainer `6.0.0`
- CuPy `6.0.0`
- ChainerCV `0.13.0`

## Weight Conversion

Converted weight for SiamMask can be downloaded from here.

- [Weight of SiamMask](https://drive.google.com/file/d/1hlz9hIlB3D-G6AgGjpK3owILtIaHrcS9/view?usp=sharing)
- [Weight of SiamRPN++](https://drive.google.com/file/d/1NLhi-tbAyJX8bJeI7Rfwwo_-BxgwPbKw/view?usp=sharing)


```bash
# Download https://github.com/STVIR/pysot
export PYSOT_DIR=$HOME/projects/pysot  # CHANGE
python pth2npz.py --snapshot $PYSOT_DIR/experiments/siammask_r50_l3/model.pth --mask
python pth2npz.py --snapshot $PYSOT_DIR/experiments/siamrpn_r50_l234_dwxcorr/model.pth
```

## Demo

### Setup
Run commands in `data/README.md`

### Run

```bash
python demo_vot.py [--pretrained-model PATH] [--gpu GPU] [--mask] [--video-id VIDEO_ID]
```

## Evaluation

```bash
python eval_sot.py [--pretrained-model PATH]  [--gpu GPU] [--mask]
```

### Evaluation result on VOT2018
The value is displayed as `Our score (Reference score)`.

| Model | EAO | Accuracy | Robustness |
|:-:|:-:|:-:|:-:|
| siamrpn_r50_l234_dwxcorr | 0.388 (0.415) | 0.599 (0.601) | 0.272 (0.234) |
| siammask_r50_l3 | 0.390 (0.423) | 0.599 (0.615) | 0.267 (0.248) |

The accuracy for SiamMask is lower than the original implementation because the evaluation measure is slightly different.
The original implementation uses polygons predicted by SiamMask for the calculation of overlap between prediction and GT.
The current implementation only uses the axis-aligned bounding boxes for the calculation of overlap.


## Test
### Test evaluation function

Download data for `test_eval_vot.py`
- [eval_siamrpn_r50_l234_dwxcorr.npz](https://drive.google.com/file/d/1SY7xZNqRRWPQW90iB__jzTGCLrdSm2fZ/view?usp=sharing)
- [all_tracker_trajs.npz](https://drive.google.com/file/d/1o0q41SKl8m4TTwgXTDWn-F32c-jdbN9h/view?usp=sharing)


### Test consistency with PyTorch model
You need pysot installed.

data:
- [template.npy](https://drive.google.com/file/d/1g6MbIsnnxIW2dHcBfFpsK30Q4Up-e7UK/view?usp=sharing)
- [track.npy](https://drive.google.com/file/d/1jgv69xzLVwE7TGfbmr1fy-vzmDnjJIDQ/view?usp=sharing)

##### Summary
NN part is almost completely the same between the two implementations (diff is smaller than 1e-5).
The other part is usually consistent.
One exception is the discretization error introduced when cropping an image based on the predicted bbox.

```bash
export PYSOT_DIR=$HOME/projects/pysot  # CHANGE

# download template.npy and track.npy
python test_model_consistency.py --snapshot $PYSOT_DIR/experiments/siammask_r50_l3/model.pth --config $PYSOT_DIR/experiments/siammask_r50_l3/config.yaml --pretrained-model chainer_model.npz --mask

python test_tracker_consistency.py --snapshot $PYSOT_DIR/experiments/siammask_r50_l3/model.pth --config $PYSOT_DIR/experiments/siammask_r50_l3/config.yaml --pretrained-model chainer_model.npz --mask
python test_tracker_consistency.py --snapshot $PYSOT_DIR/experiments/siamrpn_r50_l234_dwxcorr/model.pth --config $PYSOT_DIR/experiments/siamrpn_r50_l234_dwxcorr/config.yaml --pretrained-model chainer_model.npz
```


## Status

Inference
- [x] SiamRPN++
- [x] SiamMask
- [ ] SiamRPN + LT

Training
- [ ] SiamRPN++
- [ ] SiamMask
- [ ] SiamRPN + LT

Evaluation
- [x] VOT2018

