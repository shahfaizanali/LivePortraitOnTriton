# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: face_analysis_model.py

import numpy as np
from insightface.app.common import Face
import cv2
from .new_predictor import get_predictor
from ..utils import face_align

def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]),
                      reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2] + face['bbox'][0]) / 2 - face_center[0]) ** 2 + (
                (face['bbox'][3] + face['bbox'][1]) / 2 - face_center[1]) ** 2) ** 0.5)
    return faces

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class FaceAnalysisModel:
    def __init__(self, **kwargs):
        """
        Initializes the FaceAnalysisModel with Triton predictors for face detection and face pose estimation.
        
        Expected keyword arguments:
            - model_name: List of model identifiers or names for Triton.
            - triton_url: URL of the Triton Inference Server (default: 'localhost:8000').
            - debug: Boolean flag to enable debug mode (default: False).
        """
        self.model_names = kwargs.get("model_name", ["retinaface_det_static", "face_2dpose_106_static"])
        self.triton_url = kwargs.get("triton_url", "localhost:8000")
        self.debug = kwargs.get("debug", False)
        
        assert self.model_names, "At least one model name must be provided."
        assert len(self.model_names) >= 2, "Please provide both face detection and face pose model names."
        
        # Initialize Triton predictors
        self.face_det = get_predictor(model_name=self.model_names[0], url=self.triton_url, protocol="http", debug=self.debug)
        self.face_pose = get_predictor(model_name=self.model_names[1], url=self.triton_url, protocol="http", debug=self.debug)
        
        # Face detection parameters
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self.input_size = (512, 512)
        
        output_length = len(self.face_det.output_spec())
        if output_length == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif output_length == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif output_length == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif output_length == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

        self.lmk_dim = 2
        self.lmk_num = 212 // self.lmk_dim

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect_face(self, img):
        """
        Detects faces in an image using the Triton-based face detection model.
        
        Args:
            img (np.ndarray): Input image in BGR format.
        
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Detected bounding boxes and keypoints.
        """
        im_ratio = float(img.shape[0]) / img.shape[1]
        input_size = self.input_size
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size_tuple = tuple(img.shape[0:2][::-1])

        det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        det_img_normalized = (det_img_rgb.astype(np.float32) - self.input_mean) / self.input_std
        det_img_transposed = np.transpose(det_img_normalized, (2, 0, 1))
        det_img_batch = np.expand_dims(det_img_transposed, axis=0).astype(np.float32)

        # Prepare input dictionary for Triton
        input_spec = self.face_det.input_spec()
        if len(input_spec) != 1:
            raise ValueError("Face detection model expects exactly one input tensor.")
        input_name, _, _ = input_spec[0]
        input_dict = {
            input_name: det_img_batch  # Assuming single input
        }

        # Perform inference
        preds_dict = self.face_det.predict(input_dict)

        # Extract outputs based on output names
        # Replace 'output_name1', 'output_name2', etc., with your actual Triton output names
        outs = []
        for out_spec in self.face_det.output_spec():
            output_name = out_spec[0]
            if output_name in preds_dict:
                outs.append(preds_dict[output_name])
                if self.debug:
                    print(f"Output '{output_name}' retrieved with shape {preds_dict[output_name].shape}")
            else:
                raise ValueError(f"Expected output '{output_name}' not found in Triton response.")
        
        # Ensure you have the correct number of outputs
        expected_num_outputs = 9  # Adjust based on your model's actual outputs
        if len(outs) < expected_num_outputs:
            raise ValueError(f"Insufficient outputs received from Triton for face detection. Expected {expected_num_outputs}, got {len(outs)}.")

        o448, o471, o494, o451, o474, o497, o454, o477, o500 = outs[:expected_num_outputs]  # Adjust based on actual output names

        faces_det = [o448, o471, o494, o451, o474, o497, o454, o477, o500]
        input_height = det_img.shape[1]
        input_width = det_img.shape[2]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = faces_det[idx]
            bbox_preds = faces_det[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = faces_det[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # Generate anchor centers
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        return det, kpss

    def estimate_face_pose(self, img, face):
        """
        Estimates the facial landmarks for a detected face using the Triton-based face pose estimation model.
        
        Args:
            img (np.ndarray): Original input image in BGR format.
            face (Face): A Face object containing bounding box information.
        
        Returns:
            np.ndarray: Estimated facial landmarks.
        """
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        input_size = (192, 192)
        _scale = input_size[0] / (max(w, h) * 1.5)
        aimg, M = face_align.transform(img, center, input_size[0], _scale, rotate)
        input_size_tuple = tuple(aimg.shape[0:2][::-1])

        aimg_rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
        aimg_normalized = (aimg_rgb.astype(np.float32) - self.input_mean) / self.input_std
        aimg_transposed = np.transpose(aimg_normalized, (2, 0, 1))
        aimg_batch = np.expand_dims(aimg_transposed, axis=0).astype(np.float32)

        # Prepare input dictionary for Triton
        pose_input_spec = self.face_pose.input_spec()
        if len(pose_input_spec) != 1:
            raise ValueError("Face pose estimation model expects exactly one input tensor.")
        pose_input_name, _, _ = pose_input_spec[0]
        input_dict = {
            pose_input_name: aimg_batch  # Assuming single input
        }

        # Perform inference
        preds_dict = self.face_pose.predict(input_dict)

        # Extract output based on output names
        # Replace 'pose_output_name' with your actual Triton output name
        pred = None
        for out_spec in self.face_pose.output_spec():
            output_name = out_spec[0]
            if output_name in preds_dict:
                pred = preds_dict[output_name]
                if self.debug:
                    print(f"Pose Output '{output_name}' retrieved with shape {pred.shape}")
            else:
                raise ValueError(f"Expected output '{output_name}' not found in Triton response.")
            break  # Only take the first output

        if pred is None:
            raise ValueError("No output received from the pose estimation model.")

        pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size_tuple[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (input_size_tuple[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = face_align.trans_points(pred, IM)
        face.landmark = pred
        return pred

    def predict(self, img):
        """
        Detects faces and estimates their landmarks in the given image.
        
        Args:
            img (np.ndarray): Input image in BGR format.
        
        Returns:
            List[np.ndarray]: List of landmarks for each detected face.
        """
        dets, kpss = self.detect_face(img)
        if dets.shape[0] == 0:
            return []
        
        ret = []
        for i in range(dets.shape[0]):
            bbox = dets[i, 0:4]
            det_score = dets[i, 4]
            kps = kpss[i] if self.use_kps else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.estimate_face_pose(img, face)
            ret.append(face)
        
        ret = sort_by_direction(ret, 'large-small', None)
        outs = [x.landmark for x in ret]
        return outs

    def __del__(self):
        # Safely attempt to delete predictors if they exist
        if hasattr(self, 'face_det'):
            del self.face_det
        if hasattr(self, 'face_pose'):
            del self.face_pose
