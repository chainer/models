import numpy as np
import chainer
import cv2
import chainer.functions as F

from .copy_from_pysot.anchor import Anchors


class SiamRPNTracker(chainer.Chain):

    track_instance_size = 255
    track_exemplar_size = 127
    track_base_size = 8
    track_context_amount = 0.5
    track_penalty_k = 0.05
    track_window_influence = 0.42
    track_lr = 0.38

    anchor_stride = 8
    anchor_ratios = [0.33, 0.5, 1, 2, 3]
    anchor_scales = [8]

    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        with self.init_scope():
            self.model = model

        self.score_size = (self.track_instance_size - self.track_exemplar_size) // self.anchor_stride + 1 + self.track_base_size
        self.n_anchor = len(self.anchor_ratios) * len(self.anchor_scales)

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)

        self.window = np.tile(window.flatten(), self.n_anchor)
        self.anchors = self.generate_anchor(self.score_size)

        self.center_pos = None
        self.size = None 

    def generate_anchor(self, score_size):
        anchors = Anchors(self.anchor_stride,
                          self.anchor_ratios,
                          self.anchor_scales)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.transpose((1, 2, 3, 0)).reshape((4, -1))
        # delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        # delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.transpose((1, 2, 3, 0)).reshape((2, -1)).transpose((1, 0))
        # score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, axis=1)[:, 1].data
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width ,boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(
        self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1), 
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1), 
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        # im_patch = torch.from_numpy(im_patch)
        # if cfg.CUDA:
        #     im_patch = im_patch.cuda()
        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        if len(bbox.shape) == 2:
            bbox = bbox[0]
            y_min, x_min, y_max, x_max = bbox
            h = y_max - y_min
            w = x_max - x_min
            bbox = [x_min, y_min, w, h]

        # These values get updated later
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
       
        # calculate z crop size
        w_z = self.size[0] + self.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.track_context_amount * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img , axis=(0,1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, 
                self.track_exemplar_size, s_z, self.channel_average)

        self.model.template(self.xp.array(z_crop))

    def _track_bbox(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

        w_z = self.size[0] + self.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.track_context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.track_exemplar_size / s_z
        s_x = s_z * (self.track_instance_size / self.track_exemplar_size)
        x_crop = self.get_subwindow(
            img, self.center_pos, self.track_instance_size,
            round(s_x), self.channel_average)

        conf, loc = self.model.track(self.xp.array(x_crop))[:2]
        conf = chainer.cuda.to_cpu(conf.data)
        loc = chainer.cuda.to_cpu(loc.data)
        score = self._convert_score(conf)
        pred_bbox = self._convert_bbox(loc, self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
            
        # scale penalty
        s_c = change(sz(pred_bbox[2,:], pred_bbox[3,:]) / 
                (sz(self.size[0]*scale_z, self.size[1]*scale_z))) 

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2,:]/pred_bbox[3,:]))
        penalty = np.exp(-(r_c * s_c - 1) * self.track_penalty_k)
        pscore = penalty * score

        # window penalty 
        pscore = pscore * (1 - self.track_window_influence) + \
                self.window * self.track_window_influence
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.track_lr
        
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
   
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        y_min = cy - height / 2
        x_min = cx - width / 2
        y_max = y_min + height
        x_max = x_min + width
        bbox = np.array([[y_min, x_min, y_max, x_max]], dtype=np.float32)
        return bbox, best_score, best_idx

    def track(self, img):
        return self._track_bbox(img)[:2]
