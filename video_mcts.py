"""
Monte Carlo Tree Search (MCTS) for Video Generation with Custom Validator

This module implements a Monte Carlo Tree Search algorithm for exploring different
video generation possibilities with the Wan2.1 I2V-14B-480P model. It includes
a custom validator to evaluate the quality of generated videos.

The validator includes multiple evaluation metrics:
1. Motion: Evaluates the amount and consistency of motion in the video
2. Consistency: Measures temporal consistency between frames
3. Quality: Assesses visual quality using image sharpness
4. Physics: Evaluates physical plausibility of motion using optical flow analysis

The physics-based validator uses PhyCoBench concepts to analyze:
- Flow consistency: How smoothly motion changes between frames
- Acceleration consistency: How natural the changes in motion are
- Prediction accuracy: How well future frames can be predicted from past frames

This implementation is inspired by PhyCoBench (https://github.com/Jeckinchen/PhyCoBench),
a benchmark designed to assess the Physical Coherence of generated videos.

This implementation allows for systematic exploration of video generation parameters
to find the optimal settings for high-quality, physically plausible videos.
"""

import os
import math
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import torch.nn.functional as F
from collections import defaultdict
import logging
import sys
import cv2
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Import the PhyCo validator
from phyco_validator import PhyCoValidator

# Import Wan2.1 modules
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class VideoCacheManager:
    """
    Manages caching of generated videos to avoid regenerating the same videos.
    """
    def __init__(self, cache_dir="./video_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self):
        """Load the cache index from disk."""
        if self.cache_index_path.exists():
            try:
                with open(self.cache_index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}. Creating new index.")
        return {}

    def _save_cache_index(self):
        """Save the cache index to disk."""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)

    def _compute_state_hash(self, state):
        """Compute a hash for the state to use as a cache key."""
        # Create a deterministic string representation of the state
        # Exclude the image from hashing as it's a complex object
        state_copy = dict(state)
        if 'image' in state_copy:
            # For images, use the image path if available, otherwise exclude
            if 'image_path' in state_copy:
                state_copy['image'] = state_copy['image_path']
            else:
                del state_copy['image']

        # Sort keys for deterministic ordering
        state_str = json.dumps(state_copy, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()

    def get_cached_video(self, state):
        """Get a cached video if it exists."""
        state_hash = self._compute_state_hash(state)
        if state_hash in self.cache_index:
            cache_entry = self.cache_index[state_hash]
            video_path = self.cache_dir / cache_entry['video_path']
            if video_path.exists():
                logger.info(f"Cache hit: Using cached video for state {state_hash}")
                return cache_entry['score'], video_path, cache_entry.get('last_frame_path')
            else:
                # Video file doesn't exist, remove from cache
                logger.warning(f"Cache entry exists but video file missing: {video_path}")
                del self.cache_index[state_hash]
                self._save_cache_index()
        return None, None, None

    def cache_video(self, state, video_path, score, last_frame_path=None):
        """Cache a video and its metadata."""
        state_hash = self._compute_state_hash(state)

        # Copy the video to the cache directory if it's not already there
        if not video_path.startswith(str(self.cache_dir)):
            cached_video_path = self.cache_dir / Path(video_path).name
            try:
                import shutil
                shutil.copy2(video_path, cached_video_path)
                video_path = str(cached_video_path)
            except Exception as e:
                logger.warning(f"Error copying video to cache: {e}")

        # Store the cache entry
        self.cache_index[state_hash] = {
            'video_path': Path(video_path).name,
            'score': score,
            'timestamp': time.time(),
            'last_frame_path': last_frame_path
        }
        self._save_cache_index()
        logger.info(f"Cached video for state {state_hash}")


class VideoNode:
    """
    Node in the MCTS tree representing a state in the video generation process.
    """
    def __init__(self,
                 parent=None,
                 action=None,
                 state=None,
                 depth=0,
                 model=None,
                 cache_manager=None):
        self.parent = parent
        self.action = action  # Action that led to this state
        self.state = state    # Current state (parameters for video generation)
        self.depth = depth    # Depth in the tree
        self.children = []    # Child nodes
        self.visits = 0       # Number of visits to this node
        self.value = 0.0      # Value of this node
        self.model = model    # Reference to the model for generation
        self.video = None     # Generated video tensor
        self.video_path = None  # Path to saved video
        self.score = None     # Validation score
        self.multi_gpu = False  # Whether to use multiple GPUs
        self.last_frame = None  # Last frame of the generated video
        self.chain_videos = []  # List of chained video paths
        self.cache_manager = cache_manager  # Reference to the cache manager

    def add_child(self, action, state):
        """Add a child node with the given action and state."""
        child = VideoNode(
            parent=self,
            action=action,
            state=state,
            depth=self.depth + 1,
            model=self.model,
            cache_manager=self.cache_manager
        )
        child.multi_gpu = self.multi_gpu
        self.children.append(child)
        return child

    def update(self, value):
        """Update node statistics with a new value."""
        self.visits += 1
        self.value += (value - self.value) / self.visits

    def is_fully_expanded(self):
        """Check if all possible actions from this state have been tried."""
        return len(self.children) == len(self.get_possible_actions())

    def best_child(self, exploration_weight=1.0):
        """
        Select the best child node according to UCB1 formula.

        Args:
            exploration_weight: Weight for exploration term in UCB1

        Returns:
            The best child node
        """
        if not self.children:
            return None

        # UCB1 formula: value + exploration_weight * sqrt(ln(parent_visits) / child_visits)
        def ucb_score(child):
            exploitation = child.value
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration

        return max(self.children, key=ucb_score)

    def get_possible_actions(self):
        """
        Get all possible actions from the current state.

        Returns:
            List of possible actions
        """
        # Define the action space based on the current state
        # For video generation, actions could be changes to parameters like:
        # - seed
        # - guidance scale
        # - sampling steps
        # - shift scale
        # - prompt

        actions = []

        # Vary seed
        for seed_delta in [-100, -50, 0, 50, 100]:
            new_seed = self.state.get('seed', 42) + seed_delta
            if new_seed < 0:
                continue
            actions.append(('seed', new_seed))

        # Vary guidance scale
        for guide_scale in [3.0, 4.0, 5.0, 6.0, 7.0]:
            if abs(guide_scale - self.state.get('guide_scale', 5.0)) <= 2.0:
                actions.append(('guide_scale', guide_scale))

        # Vary sampling steps
        for steps in [20, 30, 40, 50]:
            actions.append(('sampling_steps', steps))

        # Vary shift scale
        for shift in [3.0, 4.0, 5.0, 6.0]:
            actions.append(('shift_scale', shift))

        # Use level-specific prompts if available
        if 'level_prompts' in self.state and self.state['level_prompts']:
            # Get prompts for the current depth level
            current_level = self.depth + 1  # +1 because children will be at next level
            if current_level in self.state['level_prompts']:
                level_prompts = self.state['level_prompts'][current_level]
                for prompt in level_prompts:
                    actions.append(('prompt', prompt))

        # Use alternative prompts if available and no level-specific prompts found
        elif 'alternative_prompts' in self.state and self.state['alternative_prompts']:
            for i, alt_prompt in enumerate(self.state['alternative_prompts']):
                actions.append(('prompt', alt_prompt))

        return actions

    def generate_video(self, use_last_frame=False, num_segments=1):
        """
        Generate a video using the current state parameters.

        Args:
            use_last_frame: Whether to use the last frame of the parent's video as input
            num_segments: Number of video segments to generate in sequence

        Returns:
            Tuple of (video_tensor, video_path)
        """
        if self.video is not None and not use_last_frame:
            return self.video, self.video_path

        # Check if this video is already cached
        if self.cache_manager is not None:
            # Create a copy of the state for caching
            cache_state = dict(self.state)

            # Add parent information for videos that use the last frame
            if use_last_frame and self.parent and self.parent.video_path:
                cache_state['parent_video'] = self.parent.video_path
                cache_state['use_last_frame'] = True

            # Add number of segments to the cache state
            cache_state['num_segments'] = num_segments

            # Try to get the video from cache
            cached_score, cached_video_path, cached_last_frame_path = self.cache_manager.get_cached_video(cache_state)
            if cached_video_path is not None:
                logger.info(f"Using cached video: {cached_video_path}")
                self.video_path = str(cached_video_path)
                self.score = cached_score

                # Load the video tensor if needed
                if cached_last_frame_path and Path(cached_last_frame_path).exists():
                    self.last_frame = Image.open(cached_last_frame_path)
                    logger.info(f"Loaded cached last frame: {cached_last_frame_path}")

                # Load the video tensor for validation if needed
                self.video = self._load_video_tensor(self.video_path)

                return self.video, self.video_path

        # Extract parameters from state
        prompt = self.state.get('prompt', '')
        guide_scale = self.state.get('guide_scale', 5.0)
        shift_scale = self.state.get('shift_scale', 5.0)
        sampling_steps = self.state.get('sampling_steps', 40)
        seed = self.state.get('seed', 42)
        n_prompt = self.state.get('n_prompt', '')

        # Determine the input image
        if use_last_frame and self.parent and self.parent.last_frame is not None:
            # Use the last frame of the parent's video as input
            image = self.parent.last_frame
            logger.info("Using last frame of parent video as input")
        else:
            # Use the original image from the state
            image = self.state.get('image', None)

        # Generate unique filename based on parameters
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"mcts_video_{timestamp}_{seed}.mp4"

        # Free up CUDA memory
        torch.cuda.empty_cache()

        # Initialize variables for chained video generation
        all_videos = []
        all_video_paths = []
        current_image = image

        try:
            # Generate multiple video segments in sequence
            for segment in range(num_segments):
                # Generate the video segment
                logger.info(f"Generating video segment {segment+1}/{num_segments} with parameters: {self.state}")

                # Resize image to save memory if it's too large
                if isinstance(current_image, Image.Image):
                    w, h = current_image.size
                    if w > 512 or h > 512:
                        # Resize while maintaining aspect ratio
                        if w > h:
                            new_w, new_h = 512, int(h * 512 / w)
                        else:
                            new_w, new_h = int(w * 512 / h), 512
                        current_image = current_image.resize((new_w, new_h), Image.LANCZOS)
                        logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} to save memory")

                # Set additional parameters for multi-GPU
                kwargs = {
                    'offload_model': True
                }

                if self.multi_gpu:
                    kwargs['ulysses_size'] = 4  # Use all 4 GPUs

                # Get custom resolution if specified
                max_area = MAX_AREA_CONFIGS['480*832']
                if 'height' in self.state and 'width' in self.state:
                    h = self.state.get('height')
                    w = self.state.get('width')
                    max_area = (h, w)
                    logger.info(f"Using custom resolution: {h}x{w}")

                # Generate the video segment
                segment_video = self.model.generate(
                    prompt,
                    current_image,
                    max_area=max_area,
                    shift=shift_scale,
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                    n_prompt=n_prompt,
                    seed=seed + segment,  # Use different seed for each segment
                    **kwargs
                )

                # Save the video segment
                segment_filename = f"mcts_video_{timestamp}_{seed}_segment{segment+1}.mp4"
                segment_path = cache_video(
                    tensor=segment_video[None],
                    save_file=segment_filename,
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )

                # Extract the last frame of the video segment to use as input for the next segment
                last_frame_tensor = segment_video[:, -1]  # Shape: [C, H, W]

                # Convert tensor to PIL Image for the next iteration
                last_frame_np = (last_frame_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
                last_frame_np = last_frame_np.astype(np.uint8)
                current_image = Image.fromarray(last_frame_np)

                # Store the video segment
                all_videos.append(segment_video)
                all_video_paths.append(segment_path)

            # Concatenate all video segments if multiple segments were generated
            if len(all_videos) > 1:
                # Concatenate video tensors along the frame dimension
                video = torch.cat(all_videos, dim=1)

                # Save the concatenated video
                video_path = cache_video(
                    tensor=video[None],
                    save_file=video_filename,
                    fps=16,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
            else:
                # Use the single video segment
                video = all_videos[0]
                video_path = all_video_paths[0]

            # Store the results
            self.video = video
            self.video_path = video_path
            self.chain_videos = all_video_paths

            # Store the last frame for potential use by child nodes
            last_frame_tensor = video[:, -1]  # Shape: [C, H, W]
            last_frame_np = (last_frame_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
            last_frame_np = last_frame_np.astype(np.uint8)
            self.last_frame = Image.fromarray(last_frame_np)

            # Save the last frame to disk for caching
            last_frame_path = None
            if self.cache_manager is not None:
                last_frame_filename = f"last_frame_{Path(video_path).stem}.png"
                last_frame_path = str(Path(self.cache_manager.cache_dir) / last_frame_filename)
                self.last_frame.save(last_frame_path)

                # Cache the video and its metadata
                cache_state = dict(self.state)
                if use_last_frame and self.parent and self.parent.video_path:
                    cache_state['parent_video'] = self.parent.video_path
                    cache_state['use_last_frame'] = True
                cache_state['num_segments'] = num_segments
                self.cache_manager.cache_video(cache_state, video_path, self.score, last_frame_path)

            return video, video_path

        except Exception as e:
            logger.error(f"Error generating video: {e}")
            return None, None


class VideoValidator:
    """
    Custom validator for evaluating the quality of generated videos.
    """
    def __init__(self, metrics=None):
        """
        Initialize the validator with specified metrics.

        Args:
            metrics: List of metrics to use for validation
        """
        self.metrics = metrics or ['motion', 'consistency', 'quality', 'physics']

        # Initialize PhyCo validator if needed
        self.phyco_validator = PhyCoValidator(flow_method='farneback') if 'physics' in self.metrics else None

    def validate(self, video_tensor=None, video_path=None) -> Dict[str, float]:
        """
        Validate a video using multiple metrics.

        Args:
            video_tensor: PyTorch tensor of the video (C, F, H, W)
            video_path: Path to the saved video file

        Returns:
            Dictionary of scores for each metric and an overall score
        """
        if video_tensor is None and video_path is None:
            raise ValueError("Either video_tensor or video_path must be provided")

        scores = {}

        # Load video if only path is provided
        if video_tensor is None and video_path is not None:
            video_tensor = self._load_video(video_path)

        # Calculate scores for each metric
        if 'motion' in self.metrics:
            scores['motion'] = self._evaluate_motion(video_tensor)

        if 'consistency' in self.metrics:
            scores['consistency'] = self._evaluate_consistency(video_tensor)

        if 'quality' in self.metrics:
            scores['quality'] = self._evaluate_quality(video_tensor)

        if 'physics' in self.metrics and self.phyco_validator is not None:
            physics_scores = self.phyco_validator.evaluate_video(video_tensor)
            scores['physics'] = physics_scores['physical_coherence']
            scores['flow_consistency'] = physics_scores['flow_consistency']
            scores['acceleration_consistency'] = physics_scores['acceleration_consistency']
            scores['prediction_accuracy'] = physics_scores['prediction_accuracy']

        # Calculate overall score (weighted average)
        weights = {
            'motion': 0.3,
            'consistency': 0.2,
            'quality': 0.2,
            'physics': 0.3
        }

        overall_score = sum(scores[m] * weights[m] for m in scores) / sum(weights[m] for m in scores if m in scores)
        scores['overall'] = overall_score

        return scores

    def _load_video(self, video_path):
        """
        Load a video from file into a tensor.

        Args:
            video_path: Path to the video file

        Returns:
            PyTorch tensor of the video
        """
        # Use OpenCV to load the video
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to tensor and normalize to [-1, 1]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame_tensor)

        cap.release()

        if not frames:
            raise ValueError(f"Could not load any frames from {video_path}")

        # Stack frames to create video tensor (C, F, H, W)
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor

    def _load_video_tensor(self, video_path):
        """
        Load a video tensor from a file, with error handling.

        Args:
            video_path: Path to the video file

        Returns:
            PyTorch tensor of the video, or None if loading fails
        """
        try:
            return self._load_video(video_path)
        except Exception as e:
            logger.error(f"Error loading video tensor from {video_path}: {e}")
            return None

    def _evaluate_motion(self, video_tensor):
        """
        Evaluate the amount and quality of motion in the video.

        Args:
            video_tensor: PyTorch tensor of the video (C, F, H, W)

        Returns:
            Motion score between 0 and 1
        """
        # Extract frames
        frames = [video_tensor[:, i] for i in range(video_tensor.shape[1])]

        # Calculate optical flow between consecutive frames
        flow_magnitudes = []
        for i in range(len(frames) - 1):
            # Calculate frame difference as a simple proxy for motion
            diff = torch.abs(frames[i+1] - frames[i])
            mean_diff = torch.mean(diff).item()
            flow_magnitudes.append(mean_diff)

        if not flow_magnitudes:
            return 0.0

        # Calculate statistics of motion
        mean_motion = np.mean(flow_magnitudes)
        std_motion = np.std(flow_magnitudes)

        # Score based on amount and consistency of motion
        # We want some motion (not too little, not too much)
        # Ideal mean_motion around 0.1-0.2 for normalized [-1,1] frames
        motion_amount_score = 1.0 - min(1.0, abs(mean_motion - 0.15) / 0.15)

        # We want consistent motion (low std_dev relative to mean)
        motion_consistency_score = 1.0 - min(1.0, std_motion / (mean_motion + 1e-5))

        # Combine scores
        motion_score = 0.7 * motion_amount_score + 0.3 * motion_consistency_score

        return motion_score

    def _evaluate_consistency(self, video_tensor):
        """
        Evaluate the temporal consistency of the video.

        Args:
            video_tensor: PyTorch tensor of the video (C, F, H, W)

        Returns:
            Consistency score between 0 and 1
        """
        # Extract frames
        frames = [video_tensor[:, i] for i in range(video_tensor.shape[1])]

        # Calculate structural similarity between first frame and all others
        first_frame = frames[0]
        similarities = []

        for frame in frames[1:]:
            # Calculate normalized cross-correlation as a measure of structural similarity
            # Reshape frames to 1D vectors
            f1 = first_frame.flatten()
            f2 = frame.flatten()

            # Calculate cosine similarity
            similarity = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
            similarities.append(similarity)

        if not similarities:
            return 0.0

        # We want the similarity to decrease gradually, not abruptly
        # Calculate the rate of change of similarity
        similarity_changes = [abs(similarities[i] - similarities[i-1]) for i in range(1, len(similarities))]

        # Penalize large changes in similarity
        consistency_score = 1.0 - min(1.0, np.mean(similarity_changes) * 5)

        return consistency_score

    def _evaluate_quality(self, video_tensor):
        """
        Evaluate the visual quality of the video.

        Args:
            video_tensor: PyTorch tensor of the video (C, F, H, W)

        Returns:
            Quality score between 0 and 1
        """
        # Extract frames
        frames = [video_tensor[:, i] for i in range(video_tensor.shape[1])]

        # Calculate sharpness for each frame
        sharpness_scores = []
        for frame in frames:
            # Convert to numpy for OpenCV operations
            np_frame = (frame.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
            np_frame = np_frame.astype(np.uint8)

            # Convert to grayscale
            gray = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)

            # Calculate Laplacian variance as a measure of sharpness
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            # Normalize sharpness score
            # Typical values range from 0 to a few hundred
            normalized_sharpness = min(1.0, sharpness / 500.0)
            sharpness_scores.append(normalized_sharpness)

        if not sharpness_scores:
            return 0.0

        # Calculate mean sharpness
        quality_score = np.mean(sharpness_scores)

        return quality_score


class VideoMCTS:
    """
    Monte Carlo Tree Search for video generation.
    """
    def __init__(self,
                 model,
                 validator,
                 initial_state,
                 exploration_weight=1.0,
                 max_iterations=10,
                 max_depth=3,
                 use_cache=True,
                 cache_dir="./video_cache"):
        """
        Initialize the MCTS algorithm.

        Args:
            model: The video generation model
            validator: The video validator
            initial_state: Initial state for video generation
            exploration_weight: Weight for exploration in UCB1
            max_iterations: Maximum number of iterations
            max_depth: Maximum depth of the search tree
            use_cache: Whether to use video caching
            cache_dir: Directory to store cached videos
        """
        self.model = model
        self.validator = validator
        self.initial_state = initial_state
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations
        self.max_depth = max_depth

        # Initialize cache manager if caching is enabled
        self.cache_manager = VideoCacheManager(cache_dir) if use_cache else None

        # Initialize root node with cache manager
        self.root = VideoNode(state=initial_state, model=model, cache_manager=self.cache_manager)

    def search(self, use_last_frame=True, num_segments=1):
        """
        Perform the MCTS search.

        Args:
            use_last_frame: Whether to use the last frame of the parent's video as input
            num_segments: Number of video segments to generate in sequence

        Returns:
            The best node found
        """
        for _ in range(self.max_iterations):
            # Selection and expansion
            node = self._select_and_expand()

            # Simulation
            video, video_path = node.generate_video(use_last_frame=use_last_frame, num_segments=num_segments)

            # Evaluation
            if video is not None:
                scores = self.validator.validate(video_tensor=video)
                node.score = scores['overall']

                # Backpropagation
                self._backpropagate(node, node.score)

                # Free up memory by removing the video tensor
                node.video = None
                torch.cuda.empty_cache()

            logger.info(f"Iteration {_+1}/{self.max_iterations}, Best score so far: {self._get_best_score()}")

        # Return the best node
        return self._get_best_node()

    def _select_and_expand(self):
        """
        Select a node to expand using UCB1.

        Returns:
            The selected and expanded node
        """
        node = self.root

        # Selection: traverse the tree to find a node to expand
        while node.is_fully_expanded() and node.children and node.depth < self.max_depth:
            node = node.best_child(self.exploration_weight)

        # Expansion: if the node is not fully expanded, expand it
        if not node.is_fully_expanded() and node.depth < self.max_depth:
            possible_actions = node.get_possible_actions()
            tried_actions = [child.action for child in node.children]
            untried_actions = [a for a in possible_actions if a not in tried_actions]

            if untried_actions:
                action = random.choice(untried_actions)
                # Create new state by updating the current state with the action
                new_state = dict(node.state)
                new_state[action[0]] = action[1]

                # Add child node
                node = node.add_child(action, new_state)

        return node

    def _backpropagate(self, node, score):
        """
        Backpropagate the score up the tree.

        Args:
            node: The node to start backpropagation from
            score: The score to backpropagate
        """
        while node is not None:
            node.update(score)
            node = node.parent

    def _get_best_score(self):
        """
        Get the best score found so far.

        Returns:
            The best score
        """
        best_score = 0.0
        for child in self.root.children:
            if child.score is not None and child.score > best_score:
                best_score = child.score
        return best_score

    def _get_best_node(self):
        """
        Get the best node found.

        Returns:
            The best node
        """
        if not self.root.children:
            return self.root

        # Return the child with the highest score
        best_node = max(
            [child for child in self.root.children if child.score is not None],
            key=lambda x: x.score,
            default=self.root
        )

        return best_node


def run_video_mcts(
    prompt: str,
    image_path: str,
    model_path: str = "./I2V-14B-480P",
    iterations: int = 10,
    depth: int = 3,
    exploration_weight: float = 1.0,
    use_physics: bool = True,
    flow_method: str = 'farneback',
    multi_gpu: bool = False,
    height: int = 480,
    width: int = 832,
    use_last_frame: bool = True,
    num_segments: int = 1,
    use_cache: bool = True,
    cache_dir: str = "./video_cache",
    alternative_prompts_file: str = None
):
    """
    Run the MCTS algorithm for video generation.

    Args:
        prompt: Text prompt for video generation
        image_path: Path to the input image
        model_path: Path to the model directory
        iterations: Number of MCTS iterations
        depth: Maximum depth of the search tree
        exploration_weight: Weight for exploration in UCB1

    Returns:
        The best video parameters and path
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Initialize the model
    logger.info(f"Loading model from {model_path}")
    cfg = WAN_CONFIGS['i2v-14B']

    # Configure multi-GPU settings
    device_id = 0
    rank = 0
    t5_fsdp = False
    dit_fsdp = False
    use_usp = False

    if multi_gpu:
        logger.info("Using multi-GPU configuration")
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs")

        if num_gpus > 1:
            # For Wan2.1, we'll use ulysses_size instead of FSDP
            # This avoids the need for distributed training initialization
            t5_fsdp = False
            dit_fsdp = False
            use_usp = True
            logger.info(f"Using {num_gpus} GPUs with ulysses_size={num_gpus}")
        else:
            logger.warning(f"Only {num_gpus} GPU found. Multi-GPU mode requires at least 2 GPUs.")
            logger.warning("Falling back to single GPU mode")
            multi_gpu = False

    model = wan.WanI2V(
        config=cfg,
        checkpoint_dir=model_path,
        device_id=device_id,
        rank=rank,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        use_usp=use_usp,
    )

    # Initialize the validator with appropriate metrics
    metrics = ['motion', 'consistency', 'quality']
    if use_physics:
        metrics.append('physics')
    validator = VideoValidator(metrics=metrics)

    # Configure PhyCo validator if used
    if use_physics and validator.phyco_validator is not None:
        validator.phyco_validator = PhyCoValidator(flow_method=flow_method)

    # Load alternative prompts if provided
    alternative_prompts = []
    level_prompts = {}
    if alternative_prompts_file and os.path.exists(alternative_prompts_file):
        try:
            current_level = None
            with open(alternative_prompts_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        # Skip empty lines and comments
                        # Check if this is a level marker
                        if line.startswith('# Level'):
                            try:
                                # Extract level number
                                level_str = line.split('Level')[1].split('prompts')[0].strip()
                                current_level = int(level_str)
                                if current_level not in level_prompts:
                                    level_prompts[current_level] = []
                            except:
                                current_level = None
                        continue

                    # Add to level-specific prompts if we're in a level section
                    if current_level is not None:
                        level_prompts[current_level].append(line)
                    else:
                        # Otherwise add to general alternative prompts
                        alternative_prompts.append(line)

            if level_prompts:
                logger.info(f"Loaded prompts for {len(level_prompts)} tree levels from {alternative_prompts_file}")
                for level, prompts in level_prompts.items():
                    logger.info(f"  Level {level}: {len(prompts)} prompts")
            else:
                logger.info(f"Loaded {len(alternative_prompts)} alternative prompts from {alternative_prompts_file}")
        except Exception as e:
            logger.warning(f"Error loading prompts: {e}")

    # Initial state
    initial_state = {
        'prompt': prompt,
        'image': image,
        'guide_scale': 5.0,
        'shift_scale': 3.0,  # 3.0 recommended for 480P
        'sampling_steps': 40,
        'seed': random.randint(0, 100000),
        'n_prompt': "",
        'height': height,
        'width': width,
        'alternative_prompts': alternative_prompts,
        'level_prompts': level_prompts,
        'image_path': image_path  # Store the image path for caching
    }

    # Initialize MCTS
    mcts = VideoMCTS(
        model=model,
        validator=validator,
        initial_state=initial_state,
        exploration_weight=exploration_weight,
        max_iterations=iterations,
        max_depth=depth,
        use_cache=use_cache,
        cache_dir=cache_dir
    )

    # Set multi-GPU flag
    mcts.root.multi_gpu = multi_gpu

    # Run search
    logger.info("Starting MCTS search")
    best_node = mcts.search(use_last_frame=use_last_frame, num_segments=num_segments)

    # Get best parameters and video
    best_params = best_node.state
    best_video_path = best_node.video_path
    best_score = best_node.score

    logger.info(f"MCTS search completed")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score}")
    logger.info(f"Best video path: {best_video_path}")

    return best_params, best_video_path, best_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCTS for video generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="./I2V-14B-480P", help="Path to model directory")
    parser.add_argument("--iterations", type=int, default=10, help="Number of MCTS iterations")
    parser.add_argument("--depth", type=int, default=3, help="Maximum depth of search tree")
    parser.add_argument("--exploration", type=float, default=1.0, help="Exploration weight")
    parser.add_argument("--use-physics", action="store_true", default=True, help="Use physics-based validation")
    parser.add_argument("--no-physics", action="store_false", dest="use_physics", help="Disable physics-based validation")
    parser.add_argument("--flow-method", type=str, default="farneback", choices=["farneback", "deepflow"], help="Optical flow method")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs for video generation")
    parser.add_argument("--height", type=int, default=480, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=832, help="Width of the generated video")
    parser.add_argument("--use-last-frame", action="store_true", default=True, help="Use the last frame of parent video as input for child nodes")
    parser.add_argument("--no-last-frame", action="store_false", dest="use_last_frame", help="Don't use the last frame of parent video")
    parser.add_argument("--num-segments", type=int, default=1, help="Number of video segments to generate in sequence")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use video caching between runs")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache", help="Disable video caching")
    parser.add_argument("--cache-dir", type=str, default="./video_cache", help="Directory to store cached videos")
    parser.add_argument("--alternative-prompts", type=str, help="Path to a file containing alternative prompts, one per line. Can include level-specific prompts with '# Level X prompts' headers")

    args = parser.parse_args()

    best_params, best_video_path, best_score = run_video_mcts(
        prompt=args.prompt,
        image_path=args.image,
        model_path=args.model,
        iterations=args.iterations,
        depth=args.depth,
        exploration_weight=args.exploration,
        use_physics=args.use_physics,
        flow_method=args.flow_method,
        multi_gpu=args.multi_gpu,
        height=args.height,
        width=args.width,
        use_last_frame=args.use_last_frame,
        num_segments=args.num_segments,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        alternative_prompts_file=args.alternative_prompts
    )

    print(f"\nBest video generated at: {best_video_path}")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
