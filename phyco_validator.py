"""
Physics-based Validator for Video Generation

This module implements a physics-based validator inspired by PhyCoBench to evaluate
the physical coherence of generated videos. It analyzes optical flow to assess:

1. Flow consistency: How smoothly motion changes between frames
2. Acceleration consistency: How natural the changes in motion are
3. Prediction accuracy: How well future frames can be predicted from past frames

This implementation is inspired by PhyCoBench (https://github.com/Jeckinchen/PhyCoBench),
a benchmark designed to assess the Physical Coherence of generated videos.
"""

import numpy as np
import torch
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhyCoValidator:
    """
    Physics-based validator for evaluating the physical coherence of videos.
    """
    def __init__(self, flow_method='farneback'):
        """
        Initialize the PhyCo validator.

        Args:
            flow_method: Method for optical flow calculation ('farneback' or 'deepflow')
        """
        self.flow_method = flow_method
        logger.info(f"Initialized PhyCoValidator with flow method: {flow_method}")

    def validate(self, video_tensor=None, video_path=None) -> Dict[str, float]:
        """
        Validate a video using physics-based metrics.

        Args:
            video_tensor: PyTorch tensor of the video (C, F, H, W)
            video_path: Path to the saved video file

        Returns:
            Dictionary of scores for different physics metrics
        """
        if video_tensor is None and video_path is None:
            raise ValueError("Either video_tensor or video_path must be provided")

        # Load video if only path is provided
        if video_tensor is None and video_path is not None:
            video_tensor = self._load_video(video_path)

        # Convert tensor to numpy frames for OpenCV processing
        frames = self._tensor_to_frames(video_tensor)

        # Calculate optical flow for all consecutive frame pairs
        flows = self._calculate_optical_flows(frames)

        # Calculate physics-based metrics
        flow_consistency = self._evaluate_flow_consistency(flows)
        acceleration_consistency = self._evaluate_acceleration_consistency(flows)
        prediction_accuracy = self._evaluate_prediction_accuracy(frames, flows)

        # Combine metrics into overall physics score
        physics_score = (flow_consistency + acceleration_consistency + prediction_accuracy) / 3.0

        return {
            'flow_consistency': flow_consistency,
            'acceleration_consistency': acceleration_consistency,
            'prediction_accuracy': prediction_accuracy,
            'overall': physics_score
        }

    def _load_video(self, video_path):
        """
        Load a video from file into a tensor.

        Args:
            video_path: Path to the video file

        Returns:
            Video tensor (C, F, H, W)
        """
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if not frames:
                raise ValueError(f"No frames could be read from {video_path}")

            # Convert to tensor [F, H, W, C]
            frames_np = np.stack(frames)
            # Transpose to [C, F, H, W]
            frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0
            return frames_tensor

        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return None

    def _tensor_to_frames(self, video_tensor):
        """
        Convert a video tensor to a list of numpy frames.

        Args:
            video_tensor: Video tensor (C, F, H, W)

        Returns:
            List of numpy frames [F, H, W, C]
        """
        # Ensure tensor is on CPU
        if video_tensor.device.type != 'cpu':
            video_tensor = video_tensor.cpu()

        # Normalize if needed
        if video_tensor.min() < 0 or video_tensor.max() > 1:
            video_tensor = (video_tensor - video_tensor.min()) / (video_tensor.max() - video_tensor.min())

        # Convert to numpy and transpose from [C, F, H, W] to [F, H, W, C]
        frames = video_tensor.permute(1, 2, 3, 0).numpy()

        # Convert to uint8 for OpenCV
        frames = (frames * 255).astype(np.uint8)

        return frames

    def _calculate_optical_flows(self, frames):
        """
        Calculate optical flow between consecutive frames.

        Args:
            frames: List of numpy frames [F, H, W, C]

        Returns:
            List of optical flow fields
        """
        flows = []
        prev_frame = None

        for i, frame in enumerate(frames):
            if i == 0:
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                continue

            curr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if self.flow_method == 'farneback':
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            elif self.flow_method == 'deepflow':
                # This would require additional setup for DeepFlow
                # For now, fall back to Farneback
                logger.warning("DeepFlow not implemented, falling back to Farneback")
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            else:
                raise ValueError(f"Unknown flow method: {self.flow_method}")

            flows.append(flow)
            prev_frame = curr_frame

        return flows

    def _evaluate_flow_consistency(self, flows):
        """
        Evaluate the consistency of optical flow across frames.

        Args:
            flows: List of optical flow fields

        Returns:
            Flow consistency score (0-1)
        """
        if not flows:
            return 0.0

        # Calculate flow magnitude and direction for each flow field
        magnitudes = []
        directions = []

        for flow in flows:
            # Calculate magnitude and direction
            magnitude, direction = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(magnitude)
            directions.append(direction)

        # Calculate consistency of flow magnitude and direction
        magnitude_diffs = []
        direction_diffs = []

        for i in range(1, len(magnitudes)):
            # Calculate difference in magnitude
            mag_diff = np.abs(magnitudes[i] - magnitudes[i-1])
            magnitude_diffs.append(np.mean(mag_diff))

            # Calculate difference in direction (accounting for circular nature)
            dir_diff = np.minimum(
                np.abs(directions[i] - directions[i-1]),
                2 * np.pi - np.abs(directions[i] - directions[i-1])
            )
            direction_diffs.append(np.mean(dir_diff))

        # Calculate overall flow consistency
        if magnitude_diffs and direction_diffs:
            # Normalize magnitude differences
            max_mag_diff = max(magnitude_diffs) if max(magnitude_diffs) > 0 else 1.0
            norm_mag_diffs = [d / max_mag_diff for d in magnitude_diffs]

            # Direction differences are already normalized (0-π)
            norm_dir_diffs = [d / np.pi for d in direction_diffs]

            # Calculate consistency scores (lower differences = higher consistency)
            mag_consistency = 1.0 - np.mean(norm_mag_diffs)
            dir_consistency = 1.0 - np.mean(norm_dir_diffs)

            # Combine magnitude and direction consistency
            flow_consistency = 0.5 * mag_consistency + 0.5 * dir_consistency
            return float(flow_consistency)
        else:
            return 0.0

    def _evaluate_acceleration_consistency(self, flows):
        """
        Evaluate the consistency of acceleration (changes in flow) across frames.

        Args:
            flows: List of optical flow fields

        Returns:
            Acceleration consistency score (0-1)
        """
        if len(flows) < 2:
            return 0.0

        # Calculate flow differences (acceleration)
        flow_diffs = []
        for i in range(1, len(flows)):
            flow_diff = flows[i] - flows[i-1]
            flow_diffs.append(flow_diff)

        # Calculate acceleration magnitude
        acceleration_magnitudes = []
        for flow_diff in flow_diffs:
            magnitude = np.sqrt(flow_diff[..., 0]**2 + flow_diff[..., 1]**2)
            acceleration_magnitudes.append(magnitude)

        # Calculate consistency of acceleration
        if len(acceleration_magnitudes) > 1:
            acc_diffs = []
            for i in range(1, len(acceleration_magnitudes)):
                acc_diff = np.abs(acceleration_magnitudes[i] - acceleration_magnitudes[i-1])
                acc_diffs.append(np.mean(acc_diff))

            # Normalize acceleration differences
            max_acc_diff = max(acc_diffs) if max(acc_diffs) > 0 else 1.0
            norm_acc_diffs = [d / max_acc_diff for d in acc_diffs]

            # Calculate consistency score (lower differences = higher consistency)
            acceleration_consistency = 1.0 - np.mean(norm_acc_diffs)
            return float(acceleration_consistency)
        else:
            # Not enough frames to calculate acceleration consistency
            return 0.5  # Neutral score

    def _evaluate_prediction_accuracy(self, frames, flows, prediction_horizon=2):
        """
        Evaluate how well future frames can be predicted from past frames using optical flow.

        Args:
            frames: List of numpy frames
            flows: List of optical flow fields
            prediction_horizon: How many frames ahead to predict

        Returns:
            Prediction accuracy score (0-1)
        """
        if len(frames) <= prediction_horizon or not flows:
            return 0.0

        prediction_errors = []

        for i in range(len(frames) - prediction_horizon):
            # Use flow to predict future frame
            predicted_frame = self._predict_frame(frames[i], flows[i:i+prediction_horizon])
            
            # Compare with actual future frame
            actual_frame = frames[i + prediction_horizon]
            
            # Calculate prediction error
            error = np.mean(np.abs(predicted_frame.astype(np.float32) - actual_frame.astype(np.float32)))
            
            # Normalize error (assuming 8-bit images with values 0-255)
            normalized_error = error / 255.0
            prediction_errors.append(normalized_error)

        # Calculate prediction accuracy (lower error = higher accuracy)
        if prediction_errors:
            prediction_accuracy = 1.0 - np.mean(prediction_errors)
            return float(prediction_accuracy)
        else:
            return 0.0

    def _predict_frame(self, frame, flows):
        """
        Predict a future frame using optical flow.

        Args:
            frame: Starting frame
            flows: List of optical flow fields to apply

        Returns:
            Predicted frame
        """
        h, w = frame.shape[:2]
        predicted = frame.copy()

        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Apply each flow sequentially
        for flow in flows:
            # Update coordinates based on flow
            flow_x = cv2.resize(flow[..., 0], (w, h))
            flow_y = cv2.resize(flow[..., 1], (w, h))
            
            new_x = x_coords + flow_x
            new_y = y_coords + flow_y
            
            # Ensure coordinates are within bounds
            new_x = np.clip(new_x, 0, w - 1)
            new_y = np.clip(new_y, 0, h - 1)
            
            # Remap the image using the new coordinates
            predicted = cv2.remap(predicted, new_x, new_y, cv2.INTER_LINEAR)

        return predicted
