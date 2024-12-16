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
        return sorted(faces, key=lambda face: face.bbox[0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face.bbox[0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face.bbox[1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face.bbox[1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
                      reverse=True)
    if direction == 'distance-from-retarget-face' and face_center is not None:
        return sorted(faces, key=lambda face: np.linalg.norm(np.array(face.bbox[:2]) - np.array(face_center)))
    return faces

def distance2bbox(points, distance, max_shape=None):
    """
    Decode distance prediction to bounding box.

    Args:
        points (np.ndarray): Shape (n, 2), [x, y].
        distance (np.ndarray): Distance from the given point to 4 boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image (height, width).

    Returns:
        np.ndarray: Decoded bounding boxes.
    """
    if distance.ndim != 2 or distance.shape[1] < 4:
        raise ValueError(f"Expected distance to have shape (n, >=4), got {distance.shape}")

    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])

    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """
    Decode distance prediction to keypoints.

    Args:
        points (np.ndarray): Shape (n, 2), [x, y].
        distance (np.ndarray): Distance from the given point to keypoints.
        max_shape (tuple): Shape of the image (height, width).

    Returns:
        np.ndarray: Decoded keypoints.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class FaceAnalysisModel:
    def __init__(self, **kwargs):
        """
        Initializes the FaceAnalysisModel with Triton predictors for face detection and face pose estimation.

        Expected keyword arguments:
            - model_name: List of model identifiers or names for Triton (e.g., ["face_detection_model", "face_pose_model"]).
            - triton_url: URL of the Triton Inference Server (default: 'localhost:8000').
            - debug: Boolean flag to enable debug mode (default: False).
        """
        self.model_names = kwargs.get("model_name", ["retinaface_det_static", "face_2dpose_106_static"])
        self.triton_url = kwargs.get("triton_url", "localhost:8000")
        self.debug = kwargs.get("debug", True)

        assert self.model_names, "At least two model names must be provided (face detection and face pose)."
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
        # Replace 'scores' and 'bbox_preds' with your actual Triton output names
        outs = []
        for out_spec in self.face_det.output_spec():
            output_name = out_spec[0]
            if output_name in preds_dict:
                outs.append(preds_dict[output_name])
                if self.debug:
                    print(f"Output '{output_name}' retrieved with shape {preds_dict[output_name].shape}")
            else:
                raise ValueError(f"Expected output '{output_name}' not found in Triton response.")

        # Debug: Print all outputs
        if self.debug:
            for idx, output in enumerate(outs):
                print(f"Output {idx}: Shape {output.shape}")

        # Example mapping based on expected outputs
        output_dict = {spec[0]: outs[idx] for idx, spec in enumerate(self.face_det.output_spec())}

        # Extract specific outputs
        # Ensure these names match your Triton model's output tensor names
        scores = output_dict.get('scores')       # Replace with actual output name
        bbox_preds = output_dict.get('bbox_preds')  # Replace with actual output name
        kps_preds = output_dict.get('kps_preds') if self.use_kps else None

        # Verify shapes
        if scores is None or bbox_preds is None:
            raise ValueError("Missing required outputs from Triton face detection model.")

        if bbox_preds.shape[-1] != 4:
            raise ValueError(f"Expected bbox_preds to have 4 columns, got {bbox_preds.shape[-1]}")

        # Reshape if necessary
        scores = scores.reshape(-1)  # Flatten batch and anchors
        bbox_preds = bbox_preds.reshape(-1, 4)  # Flatten batch and anchors

        # Generate anchor centers based on stride and feature map dimensions
        anchor_centers = self.generate_anchor_centers(bbox_preds.shape[0])

        # Decode bounding boxes
        bboxes = distance2bbox(anchor_centers, bbox_preds, max_shape=(img.shape[0], img.shape[1]))

        # Assuming scores correspond to the confidence of each bbox
        dets = np.hstack((bboxes, scores[:, np.newaxis])).astype(np.float32)

        # Apply Non-Maximum Suppression (NMS)
        keep = self.nms(dets)
        det = dets[keep, :]

        if self.use_kps and kps_preds is not None:
            kps_preds = kps_preds.reshape(-1, kps_preds.shape[-1] // 2, 2)
            kpss = distance2kps(anchor_centers, kps_preds, max_shape=(img.shape[0], img.shape[1]))
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        return det, kpss

    def generate_anchor_centers(self, num_anchors):
        """
        Generates anchor centers based on the feature map strides and grid sizes.

        Args:
            num_anchors (int): Total number of anchor centers across all feature maps.

        Returns:
            np.ndarray: Array of anchor centers with shape (num_anchors, 2).
        """
        anchor_centers = []
        for stride in self._feat_stride_fpn:
            feature_map_size = self.input_size[0] // stride
            grid_x, grid_y = np.meshgrid(np.arange(feature_map_size), np.arange(feature_map_size))
            centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)
            centers *= stride
            anchor_centers.append(centers)
        anchor_centers = np.vstack(anchor_centers)
        return anchor_centers

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
