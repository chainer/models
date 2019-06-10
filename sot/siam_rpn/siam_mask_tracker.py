import cv2
import numpy as np

import chainer
import chainer.functions as F

from .siam_rpn_tracker import SiamRPNTracker


def cxy_wh_2_rect(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 0-index
    """
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])


def _crop_back(image, bbox, out_sz, padding=0):
    a = (out_sz[0] - 1) / bbox[2]
    b = (out_sz[1] - 1) / bbox[3]
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)
    return crop


def _mask_post_processing(mask, center_pos, size, track_mask_threshold):
    target_mask = (mask > track_mask_threshold)
    target_mask = target_mask.astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(
                target_mask,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)] 
        polygon = contour.reshape(-1, 2)
        prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
        rbox_in_img = prbox
    else:  # empty mask
        location = cxy_wh_2_rect(center_pos, size)
        rbox_in_img = np.array([[location[0], location[1]],
                    [location[0] + location[2], location[1]],
                    [location[0] + location[2], location[1] + location[3]],
                    [location[0], location[1] + location[3]]])
    return rbox_in_img


class SiamMaskTracker(SiamRPNTracker):

    track_lr = 0.32
    track_window_influence = 0.41
    track_penalty_k = 0.1

    track_mask_output_size = 127
    track_mask_threshold = 0.15

    def track(self, img):
        _, H, W = img.shape
        # This has to be done before _track_box
        w_z = self.size[0] + self.track_context_amount * np.sum(self.size)
        h_z = self.size[1] + self.track_context_amount * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_x = s_z * (self.track_instance_size / self.track_exemplar_size)
        s_x = round(s_x)
        crop_box = [self.center_pos[0] - s_x / 2, 
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        bbox, best_score, best_idx = self._track_bbox(img)

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.refine_mask((delta_y, delta_x))
        mask = F.sigmoid(mask).reshape(
                self.track_mask_output_size, self.track_mask_output_size).data
        mask = chainer.cuda.to_cpu(mask)
        
        s = crop_box[2] / self.track_instance_size
        base_size = self.track_base_size
        stride = self.anchor_stride
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * self.track_exemplar_size,
                   s * self.track_exemplar_size]

        s = self.track_mask_output_size / sub_box[2]

        im_h = H
        im_w = W
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = _crop_back(mask, back_box, (im_w, im_h))

        polygon = _mask_post_processing(
                mask_in_img, self.center_pos, self.size, self.track_mask_threshold)
        polygon = polygon.flatten().tolist()
        return (
                bbox, best_score,
                mask_in_img[None] > self.track_mask_threshold, polygon)
