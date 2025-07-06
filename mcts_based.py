# Define a small class for MCTS nodes
from wan.configs import MAX_AREA_CONFIGS
import math
import random
import torch
import imageio
import os
import torch.distributed as dist
import logging
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from evaluate_window import evaluate_video
from wan.utils.utils import cache_video
import tempfile
from graphviz import Digraph





def visualize_mcts_tree(root, output_file='mcts_tree'):
    dot = Digraph(comment='MCTS Tree')

    def add_node_recursively(node):
        if node.visits > 0:
            avg_reward = node.value / node.visits
        else:
            avg_reward = 0.0
        
        label = f"{node.video_id}\nReward: {avg_reward:.2f}\nVisits: {node.visits}"
        dot.node(node.video_id, label)

        if node.parent:
            dot.edge(node.parent.video_id, node.video_id)

        for child in node.children:
            add_node_recursively(child)

    add_node_recursively(root)

    # Save to file
    dot.render(output_file, format='pdf', cleanup=True)
    print(f"[INFO] MCTS tree saved to: {output_file}.pdf")

def offload_wan_model(wan):
    wan.model.to("cpu")
    wan.clip.model.to("cpu")
    if not wan.t5_cpu:
        wan.text_encoder.model.to("cpu")
    torch.cuda.empty_cache()

def restore_wan_model(wan):
    wan.model.to("cuda")
    wan.clip.model.to("cuda")
    if not wan.t5_cpu:
        wan.text_encoder.model.to("cuda")


def save_video(frames, path):
    # 1) Permute to (T, H, W, C)
    #    dim 0 → C, 1 → T, 2 → W, 3 → H
    video = frames.permute(1, 2, 3, 0)   # now (T, H, W, C)

    # 2) Scale to [0–255], clamp, to uint8
    np_video = (video
        .mul(255)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )

    # 3) Save to file
    writer = imageio.get_writer(path, fps=16, codec="libx264")
    for img in np_video:
        writer.append_data(img)
    writer.close()

# Helper function to ensure tensors have the same spatial dimensions before concatenation
def resize_tensors_to_match(tensor_list, target_size=None):
    if not tensor_list:
        return []
    
    # If no target size provided, use the first tensor's size
    if target_size is None:
        target_size = (tensor_list[0].shape[2], tensor_list[0].shape[3])
    
    resized_tensors = []
    for tensor in tensor_list:
        # Check if resizing is needed
        if tensor.shape[2] != target_size[0] or tensor.shape[3] != target_size[1]:
            # Resize to match - preserves batch and channel dims, resizes H,W
            resized = F.interpolate(
                tensor.reshape(-1, tensor.shape[2], tensor.shape[3]).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).reshape(tensor.shape[0], tensor.shape[1], target_size[0], target_size[1])
            resized_tensors.append(resized)
        else:
            resized_tensors.append(tensor)
    
    return resized_tensors

def get_reward(video, prompts=None, weights=None, frame_num=None, node_name=None, output_path=None, sample_fps=None):
    #TODO: implement reward function
    # for every level in the tree we want a different prompt
    # we will use the evaluate_video function here, we need prompt input and video input
    # we need to average the score into 1 score, with the weights as hyperparameters 
    print(prompts)
    # If prompt is provided, we can use it for evaluation
    if prompts is not None:
        # Use the prompt in evaluation
        #with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        #    temp_path = tmpfile.name
        #    print("Saving...",temp_path)
        temp_path = f"{output_path}/rollout_{node_name}.mp4"
        #save_video(video, temp_path)  # You already have this function
        
        # Ensure video is on GPU for cache_video operation
        video_gpu = video.to(device=torch.device('cuda')) if hasattr(video, 'to') and not video.is_cuda else video
        
        cache_video(
                 tensor=video_gpu[None],
                 save_file=temp_path,
                 fps=sample_fps,
                 nrow=1,
                 normalize=True,
                 value_range=(-1, 1))
        
        # Clear GPU memory of video tensor
        del video_gpu
        torch.cuda.empty_cache()
        
        #save_video(video, temp_path) 
        
        # Evaluate
        scores = evaluate_video(temp_path, prompts, frame_num)
        print(scores)

        # Clean up
        #os.remove(temp_path)

        if weights is None:
            # Default weights for: [visual_quality, temporal_consistency, dynamic_degree, text_alignment, factual_consistency]
            weights = [0, 0, 0, 1.0, 0]
            # weights = [0.33, 0.33, 0, 0.33, 0]
        
        # Calculate weighted average 
        weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        
        # Explicitly clear CUDA cache after evaluation
        torch.cuda.empty_cache()

        return weighted_score  
        
    return 1.0

# Helper function to convert tensor to PIL Image
def tensor_to_pil(tensor):
    # Ensure tensor is on CPU
    tensor = tensor.cpu()
    # Convert from [-1, 1] to [0, 1] range
    tensor = (tensor + 1) / 2
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    return transform(tensor)


class MCTSNode:
    def __init__(self, level, video, video_id, last_frames, parent=None, action=None, prompt=None, prev_prompts=None, gen_info=None):
        self.level = level                          # current generation step (0-indexed)
        self.parent = parent                        # parent node
        # Store video on CPU to save GPU memory
        self.video = video.cpu() if video is not None and hasattr(video, 'cpu') else video
        self.video_id = video_id                    # identifier of the video whithin the tree (format: {parent.video_id}_{branch_index})
        self.last_frames = last_frames              # list of frames for conditioning (list of PIL Images)
        self.action = action                        # branch index chosen from parent
        self.children = []                          # list of child MCTSNod
        self.visits = 0                             # number of visits to the node
        self.value = 0.0                            # accumulated reward value
        self.prompt = prompt                        # current prompt used for this node
        self.prev_prompts = prev_prompts or []      # list of previous prompts in the path to this node
        self.gen_info = gen_info                    # generation info for this node


def mcts_t2i_generate(
    model,
    timesteps=3,         
    branching=2,         
    num_simulations=3,
    conditioning_frames=1,  # Number of frames to use for conditioning
    conditioning_method='uniform',  # 'last_n' or 'uniform'
    use_prompt_context=True,  # Whether to use previous prompts as context
    use_exact_token_counting=False,  # Whether to use exact token counting
    max_conditioning_frames=0,  # Maximum number of conditioning frames (0 means no limit)
    accumulate_conditioning_frames=False,  # Whether to accumulate frames from previous chunks
    evaluation_weights=None,  # Five weights for evaluation metrics
    **kwargs,
):
    # Get distributed environment variables
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    
    # Set up parameters for generation
    size = kwargs["size"]
    prompts = kwargs["prompts"]
    img = kwargs["img"]
    frame_num = kwargs["frame_num"]
    sample_shift = kwargs["sample_shift"]
    sample_solver = kwargs["sample_solver"]
    sampling_steps = kwargs["sampling_steps"]
    guide_scale = kwargs["guide_scale"]
    base_seed = kwargs["base_seed"]
    offload_model = kwargs["offload_model"]
    output_path = kwargs["output_path"]
    sample_fps = kwargs["sample_fps"]
    debug = kwargs["debug"] if rank == 0 else False

    # Validate conditioning parameters
    if conditioning_method not in ['last_n', 'uniform']:
        if rank == 0 and debug:
            print(f"[WARNING] Invalid conditioning method '{conditioning_method}', defaulting to 'last_n'")
        conditioning_method = 'last_n'
    
    if conditioning_frames < 1:
        if rank == 0 and debug:
            print(f"[WARNING] Invalid number of conditioning frames {conditioning_frames}, defaulting to 1")
        conditioning_frames = 1
    
    if rank == 0 and debug:
        print(f"[DEBUG] Using {conditioning_frames} frames for conditioning with method '{conditioning_method}'")
        if not accumulate_conditioning_frames:
            print(f"[DEBUG] Frame accumulation is DISABLED - only using frames from most recent chunk")
        if use_prompt_context:
            print(f"[DEBUG] Using previous prompts as context for generation")
            if use_exact_token_counting:
                print(f"[DEBUG] Using exact token counting with tokenizer instead of approximation")
        if max_conditioning_frames > 0:
            print(f"[DEBUG] Maximum conditioning frames limit set to {max_conditioning_frames}")
        else:
            print(f"[DEBUG] No limit on maximum conditioning frames")
        if evaluation_weights is not None:
            print(f"[DEBUG] Using custom evaluation weights: {evaluation_weights}")
        else:
            print(f"[DEBUG] Using default evaluation weights: [0, 0, 0, 1.0, 0]")
    
    # Store original image dimensions for consistent resizing
    if rank == 0:
        # create the dir of the output path if it does not exist
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        if isinstance(img, Image.Image):
            original_width, original_height = img.size
        else:
            # If it's already a tensor, get dimensions
            if len(img.shape) == 3:  # C,H,W format
                original_height, original_width = img.shape[1:3]
            else:
                # Default fallback
                original_width, original_height = 512, 512
                
        if debug:
            print(f"[DEBUG] Original image dimensions: {original_width}x{original_height}")
    else:
        original_width, original_height = None, None
        
    # Broadcast original dimensions to all processes
    if is_distributed:
        # Use a safer approach for distributed communication
        if rank == 0:
            # Create a list with the dimensions
            dim_list = [original_width, original_height]
        else:
            # Empty list on other ranks
            dim_list = [0, 0]
        
        # Use object_list broadcasting which is safer for different devices
        dist.broadcast_object_list(dim_list, src=0)
        
        # Get dimensions from the list
        if rank != 0:
            original_width, original_height = dim_list[0], dim_list[1]

    # Helper: UCB score calculation
    def ucb_score(child, parent_visits, c=1.41):
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + c * math.sqrt(math.log(parent_visits) / child.visits)

    # Helper: Resize image to original dimensions
    def resize_to_original(image):
        if isinstance(image, Image.Image):
            return image.resize((original_width, original_height), Image.LANCZOS)
        elif isinstance(image, torch.Tensor):
            # Handle tensor resizing
            if len(image.shape) == 3:  # C,H,W format
                image = image.unsqueeze(0)  # Add batch dimension
                image = F.interpolate(image, size=(original_height, original_width), mode='bilinear', align_corners=False)
                return image.squeeze(0)  # Remove batch dimension
            else:
                return image
        return image

    # Initialize root node (only on rank 0)
    if rank == 0:
        # For the initial case, we only have one conditioning frame (the input image)
        root = MCTSNode(level=0, video=None, video_id="0", last_frames=[img], parent=None, action=None, prompt=None, prev_prompts=None)
        if debug:
            print("[DEBUG] Starting MCTS with root node at level 0.")
    
    # Main MCTS loop
    for sim in range(num_simulations):
        # Step 1: Selection (rank 0 only)
        node_empty = 0
        if rank == 0:
            # Select a leaf node to expand
            current = root
            while current.level < timesteps and current.children:
                if len(current.children) < branching:
                    break
                best = None
                best_score = -float("inf")
                for child in current.children:
                    score = ucb_score(child, current.visits if current.visits > 0 else 1)
                    if score > best_score:
                        best_score = score
                        best = child
                current = best
            selected_node = current

            if selected_node.parent is not None and selected_node.video is None:
                # generate the video for the parent node
                gen_info = selected_node.gen_info
                action = selected_node.action
                node_empty = 1
         # Distribute selected_node information to all ranks if it exists
        if is_distributed:
            # First, broadcast whether selected_node exists
            node_empty_list = [node_empty]
            dist.broadcast_object_list(node_empty_list, src=0)
            if rank != 0:
                node_empty = node_empty_list[0]
            
            if node_empty == 1:
                if rank == 0:
                    # Create object list for broadcasting
                    data_to_broadcast = [gen_info, action]
                else:
                    # Non-root processes create empty placeholders
                    data_to_broadcast = [None, None]
                
                # Broadcast the data
                dist.broadcast_object_list(data_to_broadcast, src=0)
                # Non-root processes unpack the received data
                gen_info = data_to_broadcast[0]
                branch = data_to_broadcast[1]
        
        if node_empty == 1:
            # All processes generate video - note that we only use the last frame for actual generation
            # as per the requirement to only preserve the last frame's latent representation
            if len(gen_info["conditioning_frames"]) > 1:
                # Multiple conditioning frames
                if use_prompt_context and gen_info["prev_prompt"] is not None:
                    # Use both multiple frames and prompt context
                    # Set model's token counting mode to exact if requested
                    model.use_exact_token_counting = use_exact_token_counting
                    
                    video = model.generate_multi_with_context(
                        gen_info["prompt"],
                        context_prompt=gen_info["prev_prompt"],
                        imgs=gen_info["conditioning_frames"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
                else:
                    # Use only multiple frames without prompt context
                    video = model.generate_multi(
                        gen_info["prompt"],
                        imgs=gen_info["conditioning_frames"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
            else:
                # Single conditioning frame
                if use_prompt_context and gen_info["prev_prompt"] is not None:
                    # Use prompt context with single frame
                    # Set model's token counting mode to exact if requested
                    model.use_exact_token_counting = use_exact_token_counting
                    
                    video = model.generate_with_context(
                        gen_info["prompt"],
                        context_prompt=gen_info["prev_prompt"],
                        img=gen_info["image"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
                else:
                    # Original generation with single frame, no context
                    video = model.generate(
                        gen_info["prompt"],
                        img=gen_info["image"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
            if rank == 0:
                # Extract frames for conditioning based on selected method
                if conditioning_method == 'last_n':
                    new_frames = extract_last_n_frames(video, conditioning_frames)
                else:  # 'uniform'
                    new_frames = sample_uniform_frames(video, conditioning_frames)
                
                # Resize all frames to match original dimensions
                new_frames = [resize_to_original(frame) for frame in new_frames]
                
                # Combine parent's frames with new frames or use only new frames depending on setting
                if accumulate_conditioning_frames:
                    # Accumulate frames from previous chunks
                    all_conditioning_frames = list(gen_info["conditioning_frames"])  # Start with parent's frames
                    all_conditioning_frames.extend(new_frames)  # Add new frames
                else:
                    # Use only frames from current chunk (no accumulation)
                    all_conditioning_frames = new_frames
                
                # Limit the total number of conditioning frames if needed
                if max_conditioning_frames > 0 and len(all_conditioning_frames) > max_conditioning_frames:
                    if debug:
                        print(f"[DEBUG] Limiting conditioning frames from {len(all_conditioning_frames)} to {max_conditioning_frames}")
                    if conditioning_method == 'last_n':
                        # Keep the most recent frames
                        all_conditioning_frames = all_conditioning_frames[-max_conditioning_frames:]
                    else:  # 'uniform'
                        # Sample frames uniformly from all available frames
                        step_size = len(all_conditioning_frames) / max_conditioning_frames
                        indices = [int(i * step_size) for i in range(max_conditioning_frames)]
                        all_conditioning_frames = [all_conditioning_frames[i] for i in indices]
                
                child_id = gen_info["child_id"]
                
                # Save video if output path is specified
                if output_path:
                    cache_video(
                    tensor=video[None],
                    save_file=f"{output_path}/{child_id}.mp4",
                    fps=sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                
                # Explicitly move video to CPU here (even though the MCTSNode constructor will do it as well)
                # This ensures the GPU memory is freed as soon as possible
                video_cpu = video.cpu() if hasattr(video, 'cpu') else video
                selected_node.video = video_cpu
                selected_node.last_frames = all_conditioning_frames

                if debug:
                    print(f"[DEBUG] Generated video for parent node at level {selected_node.level} with ID {selected_node.video_id} has {len(selected_node.last_frames)} conditioning frames")

                # Clear GPU memory
                del video
                torch.cuda.empty_cache()

        if rank == 0:
            # Get the last frame from the selected node
            selection_info = {
                "level": selected_node.level,
                "last_frames": selected_node.last_frames,
                "video_id": selected_node.video_id
            }
            if debug:
                print(f"[DEBUG] Selected node {selected_node.video_id} at level {selected_node.level} for expansion")
        
            # Step 2: Expansion (all processes participate in generation)
            # But only rank 0 maintains the tree
            
            # For expansion, we generate multiple children from the selected node
            if selection_info["level"] < timesteps:
                for branch in range(branching):
                    # All processes generate videos with the same parameters
                    
                    current_prompt = prompts[selection_info["level"]]
                    
                    # For conditioning, use the last frame from the available frames
                    # This ensures we preserve only the last frame's latent representation
                    current_image = selection_info["last_frames"][-1]
                    
                    # Get previous prompt for context if available and requested
                    prev_prompt = None
                    if use_prompt_context and selection_info["level"] > 0:
                        if selected_node.prev_prompts and len(selected_node.prev_prompts) > 0:
                            # Pass all previous prompts rather than just the most recent one
                            prev_prompt = selected_node.prev_prompts
                    child_id = f"{selection_info['video_id']}_{branch}"
                    gen_info = {
                        "prompt": current_prompt,
                        "prev_prompt": prev_prompt,
                        "image": current_image,
                        "conditioning_frames": selection_info["last_frames"],
                        "child_id": child_id
                    }
                
                    

                    # Create child node
                    child = MCTSNode(
                        level=selected_node.level + 1, 
                        video=None,
                        video_id=child_id, 
                        last_frames=None,  # Store combined frames for conditioning
                        parent=selected_node,
                        action=branch,
                        prompt=gen_info["prompt"],
                        prev_prompts=selected_node.prev_prompts + [gen_info["prompt"]],
                        gen_info=gen_info
                    )
                    
                    
                    # Add child to selected node
                    selected_node.children.append(child)
                    
                    if debug:
                        print(f"[DEBUG] Created child node at level {child.level} with ID {child.video_id}")
            
        # Step 3: Simulation - perform a rollout from a randomly selected child
        # Only rank 0 selects which child to simulate
        has_selected_child = 0
        if rank == 0 and hasattr(selected_node, 'children') and selected_node.children:
            selected_child = random.choice(selected_node.children)
            has_selected_child = 1
            print(f"[DEBUG] Simulating child node with ID {selected_child.video_id} at level {selected_child.level} from {selected_node.video_id}")
            rollout_info = {
                "level": selected_child.level,
                "last_frames": selected_child.last_frames  # This is now a list of PIL Images
            }
            # Prepare data to broadcast
            gen_info = selected_child.gen_info
            action = selected_child.action
            
        else:
            rollout_info = None

        # Distribute selected_child information to all ranks if it exists
        if is_distributed:
            # First, broadcast whether selected_child exists
            has_selected_child_list = [has_selected_child]
            dist.broadcast_object_list(has_selected_child_list, src=0)
            if rank != 0:
                has_selected_child = has_selected_child_list[0]
            
            if has_selected_child == 1:
                if rank == 0:
                    # Create object list for broadcasting
                    data_to_broadcast = [gen_info, action]
                else:
                    # Non-root processes create empty placeholders
                    data_to_broadcast = [None, None]
                
                # Broadcast the data
                dist.broadcast_object_list(data_to_broadcast, src=0)

                # Non-root processes unpack the received data
                gen_info = data_to_broadcast[0]
                branch = data_to_broadcast[1]
        # Only rank 0 updates the tree
        if has_selected_child == 1:
            # All processes generate video - note that we only use the last frame for actual generation
            # as per the requirement to only preserve the last frame's latent representation
            if len(gen_info["conditioning_frames"]) > 1:
                # Multiple conditioning frames
                if use_prompt_context and gen_info["prev_prompt"] is not None:
                    # Use both multiple frames and prompt context
                    # Set model's token counting mode to exact if requested
                    model.use_exact_token_counting = use_exact_token_counting
                    
                    video = model.generate_multi_with_context(
                        gen_info["prompt"],
                        context_prompt=gen_info["prev_prompt"],
                        imgs=gen_info["conditioning_frames"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
                else:
                    # Use only multiple frames without prompt context
                    video = model.generate_multi(
                        gen_info["prompt"],
                        imgs=gen_info["conditioning_frames"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
            else:
                # Single conditioning frame
                if use_prompt_context and gen_info["prev_prompt"] is not None:
                    # Use prompt context with single frame
                    # Set model's token counting mode to exact if requested
                    model.use_exact_token_counting = use_exact_token_counting
                    
                    video = model.generate_with_context(
                        gen_info["prompt"],
                        context_prompt=gen_info["prev_prompt"],
                        img=gen_info["image"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
                else:
                    # Original generation with single frame, no context
                    video = model.generate(
                        gen_info["prompt"],
                        img=gen_info["image"],
                        max_area=MAX_AREA_CONFIGS[size],
                        frame_num=frame_num,
                        shift=sample_shift,
                        sample_solver=sample_solver,
                        sampling_steps=sampling_steps,
                        guide_scale=guide_scale,
                        seed=base_seed + branch,  # Use different seeds for each branch
                        offload_model=offload_model
                    )
            if rank == 0:
                # Extract frames for conditioning based on selected method
                if conditioning_method == 'last_n':
                    new_frames = extract_last_n_frames(video, conditioning_frames)
                else:  # 'uniform'
                    new_frames = sample_uniform_frames(video, conditioning_frames)
                
                # Resize all frames to match original dimensions
                new_frames = [resize_to_original(frame) for frame in new_frames]
                
                # Combine parent's frames with new frames or use only new frames depending on setting
                if accumulate_conditioning_frames:
                    # Accumulate frames from previous chunks
                    all_conditioning_frames = list(gen_info["conditioning_frames"])  # Start with parent's frames
                    all_conditioning_frames.extend(new_frames)  # Add new frames
                else:
                    # Use only frames from current chunk (no accumulation)
                    all_conditioning_frames = new_frames
                
                # Limit the total number of conditioning frames if needed
                if max_conditioning_frames > 0 and len(all_conditioning_frames) > max_conditioning_frames:
                    if debug:
                        print(f"[DEBUG] Limiting conditioning frames from {len(all_conditioning_frames)} to {max_conditioning_frames}")
                    if conditioning_method == 'last_n':
                        # Keep the most recent frames
                        all_conditioning_frames = all_conditioning_frames[-max_conditioning_frames:]
                    else:  # 'uniform'
                        # Sample frames uniformly from all available frames
                        step_size = len(all_conditioning_frames) / max_conditioning_frames
                        indices = [int(i * step_size) for i in range(max_conditioning_frames)]
                        all_conditioning_frames = [all_conditioning_frames[i] for i in indices]
                
                child_id = gen_info["child_id"]
                
                # Save video if output path is specified
                if output_path:
                    cache_video(
                    tensor=video[None],
                    save_file=f"{output_path}/{child_id}.mp4",
                    fps=sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                
                # Explicitly move video to CPU here (even though the MCTSNode constructor will do it as well)
                # This ensures the GPU memory is freed as soon as possible
                video_cpu = video.cpu() if hasattr(video, 'cpu') else video
                selected_child.video = video_cpu
                selected_child.last_frames = all_conditioning_frames

                if debug:
                    print(f"[DEBUG] Generated video for child node at level {selected_child.level} with ID {selected_child.video_id} has {len(selected_child.last_frames)} conditioning frames")

                # Clear GPU memory
                del video
                torch.cuda.empty_cache()
  

        if rank == 0 and hasattr(selected_node, 'children') and selected_node.children:
            rollout_info['last_frames'] = selected_child.last_frames

        # Broadcast rollout info
        if is_distributed:
            rollout_info_list = [rollout_info]
            dist.broadcast_object_list(rollout_info_list, src=0)
            rollout_info = rollout_info_list[0]
            
        # Perform rollout - all processes participate in generation
        video_segments = []
        video_prompts = []
        
        # Generate remaining segments
        if rollout_info and rollout_info["level"] < timesteps:

            # Use the last frame from the conditioning frames for generation
            current_frame = rollout_info["last_frames"][-1]  # This is a PIL Image
            current_conditioning_frames = rollout_info["last_frames"]  # All frames for conditioning
            
            # Initialize a list to store frames for conditioning
            if accumulate_conditioning_frames:
                # Start with all existing conditioning frames when accumulation is enabled
                all_previous_frames = list(current_conditioning_frames)
            else:
                # When accumulation is disabled, only use the last frame from current chunk
                all_previous_frames = [current_frame]
            
            # Initialize list of previous prompts for context conditioning
            all_previous_prompts = []
            if rank == 0 and selected_child and selected_child.prev_prompts:
                # Start with the prompts already in the selected child node
                all_previous_prompts = list(selected_child.prev_prompts)
                
            if rank == 0:    
                print(f"[DEBUG] Starting rollout from level {rollout_info['level']} and child ID {selected_child.video_id}")

            # Generate remaining video segments
            for step in range(rollout_info["level"], timesteps):
                if rank == 0:
                    current_prompt = prompts[step]
                    
                    # Get previous prompts for context if available and requested
                    prev_prompt = None
                    if use_prompt_context and step > 0:
                        # Use all accumulated previous prompts instead of just the last one
                        if all_previous_prompts:
                            prev_prompt = all_previous_prompts
                        # If no accumulated prompts (shouldn't happen), fall back to previous step prompt
                        elif step > 0:
                            prev_prompt = [prompts[step-1]]
                        
                    rollout_step_info = {
                        "prompt": current_prompt,
                        "prev_prompt": prev_prompt,
                        "image": current_frame,  # Last frame for generation
                        "conditioning_frames": all_previous_frames  # All frames from all previous chunks
                    }
                else:
                    rollout_step_info = None
                
                # Broadcast step info
                if is_distributed:
                    rollout_step_list = [rollout_step_info]
                    dist.broadcast_object_list(rollout_step_list, src=0)
                    rollout_step_info = rollout_step_list[0]
                
                # All processes generate using only the last frame or multiple frames
                if len(rollout_step_info["conditioning_frames"]) > 1:
                    # Multiple conditioning frames
                    if use_prompt_context and rollout_step_info["prev_prompt"] is not None:
                        # Use both multiple frames and prompt context
                        # Set model's token counting mode to exact if requested
                        model.use_exact_token_counting = use_exact_token_counting
                        
                        video_segment = model.generate_multi_with_context(
                            rollout_step_info["prompt"],
                            context_prompt=rollout_step_info["prev_prompt"],
                            imgs=rollout_step_info["conditioning_frames"],
                            max_area=MAX_AREA_CONFIGS[size],
                            frame_num=frame_num,
                            shift=sample_shift,
                            sample_solver=sample_solver,
                            sampling_steps=sampling_steps,
                            guide_scale=guide_scale,
                            seed=base_seed + 1000 + step,  # Use different seeds for rollout
                            offload_model=offload_model
                        )
                    else:
                        # Use only multiple frames without prompt context
                        video_segment = model.generate_multi(
                            rollout_step_info["prompt"],
                            imgs=rollout_step_info["conditioning_frames"],
                            max_area=MAX_AREA_CONFIGS[size],
                            frame_num=frame_num,
                            shift=sample_shift,
                            sample_solver=sample_solver,
                            sampling_steps=sampling_steps,
                            guide_scale=guide_scale,
                            seed=base_seed + 1000 + step,  # Use different seeds for rollout
                            offload_model=offload_model
                        )
                else:
                    # Single conditioning frame
                    if use_prompt_context and rollout_step_info["prev_prompt"] is not None:
                        # Use prompt context with single frame
                        # Set model's token counting mode to exact if requested
                        model.use_exact_token_counting = use_exact_token_counting
                        
                        video_segment = model.generate_with_context(
                            rollout_step_info["prompt"],
                            context_prompt=rollout_step_info["prev_prompt"],
                            img=rollout_step_info["image"],
                            max_area=MAX_AREA_CONFIGS[size],
                            frame_num=frame_num,
                            shift=sample_shift,
                            sample_solver=sample_solver,
                            sampling_steps=sampling_steps,
                            guide_scale=guide_scale,
                            seed=base_seed + 1000 + step,  # Use different seeds for rollout
                            offload_model=offload_model
                        )
                    else:
                        # Original generation with single frame, no context
                        video_segment = model.generate(
                            rollout_step_info["prompt"],
                            img=rollout_step_info["image"],  # Only use last frame for generation
                            max_area=MAX_AREA_CONFIGS[size],
                            frame_num=frame_num,
                            shift=sample_shift,
                            sample_solver=sample_solver,
                            sampling_steps=sampling_steps,
                            guide_scale=guide_scale,
                            seed=base_seed + 1000 + step,  # Use different seeds for rollout
                            offload_model=offload_model
                        )
                
                # Update for next iteration
                if rank == 0:
                    # Extract frames from current video segment
                    if conditioning_method == 'last_n':
                        new_frames = extract_last_n_frames(video_segment, conditioning_frames)
                    else:  # 'uniform'
                        new_frames = sample_uniform_frames(video_segment, conditioning_frames)
                    
                    # Resize frames to maintain consistency
                    new_frames = [resize_to_original(frame) for frame in new_frames]
                    
                    # Add new frames to accumulated frames for extended conditioning based on setting
                    if accumulate_conditioning_frames:
                        # Accumulate frames from previous chunks
                        all_previous_frames.extend(new_frames)
                    else:
                        # Use only frames from current chunk (no accumulation)
                        all_previous_frames = new_frames
                    
                    # Add current prompt to accumulated prompts for next iteration
                    all_previous_prompts.append(current_prompt)
                    
                    # Limit the total number of conditioning frames if needed
                    if max_conditioning_frames > 0 and len(all_previous_frames) > max_conditioning_frames:
                        if debug:
                            print(f"[DEBUG] Limiting conditioning frames from {len(all_previous_frames)} to {max_conditioning_frames}")
                        if conditioning_method == 'last_n':
                            # Keep the most recent frames
                            all_previous_frames = all_previous_frames[-max_conditioning_frames:]
                        else:  # 'uniform'
                            # Sample frames uniformly from all available frames
                            step_size = len(all_previous_frames) / max_conditioning_frames
                            indices = [int(i * step_size) for i in range(max_conditioning_frames)]
                            all_previous_frames = [all_previous_frames[i] for i in indices]
                    
                    # Use the last frame for the next generation
                    current_frame = new_frames[-1]
                    
                    # Set current_conditioning_frames to new_frames for backward compatibility
                    current_conditioning_frames = new_frames
                    
                    # Move video segment to CPU to save GPU memory
                    video_segment_cpu = video_segment.cpu()
                    video_segments.append(video_segment_cpu)
                    video_prompts.append(rollout_step_info["prompt"])
                    
                    # Clear GPU memory
                    del video_segment
                    torch.cuda.empty_cache()
        
        # Step 4: Backpropagation (rank 0 only)
        if rank == 0: #and selected_node.children:
            # Compute reward
            # Get videos from the path to the selected child
            path_segments = []
            path_prompts = []
            if selected_node.level == timesteps:
                current = selected_node
                print(f"[DEBUG] Backprogating from full depth node {selected_node.video_id}")
            else:
                current = selected_child
                print(f"[DEBUG] Backprogating from child node {selected_child.video_id} started from {selected_node.video_id}")
            while current.parent:
                if current.video is not None:
                    # Move video to GPU for concatenation
                    path_segments.append(current.video.to(device=torch.device('cuda')))
                    path_prompts.append(current.prompt)
                current = current.parent
            path_segments.reverse()
            path_prompts.reverse()
            
            # Add rollout segments (move them back to GPU first)
            gpu_video_segments = [segment.to(device=torch.device('cuda')) for segment in video_segments]
            path_segments.extend(gpu_video_segments)
            path_prompts.extend(video_prompts)
            
            # Clear CPU video segments memory
            del video_segments
            del gpu_video_segments

            offload_wan_model(model)
            # Concatenate all video segments
            print(len(path_segments), len(path_prompts))
            if path_segments:
                # Resize all tensors to match the spatial dimensions of the first one
                # to avoid dimension mismatch errors
                # path_segments = resize_tensors_to_match(path_segments)
                full_video = torch.cat(path_segments, dim=1)
                if selected_node.level == timesteps and selected_node.visits != 0:
                    reward = selected_node.value/selected_node.visits
                else:
                    reward = get_reward(full_video, path_prompts, frame_num=frame_num, node_name=selected_child.video_id, output_path=output_path, sample_fps=sample_fps, weights=evaluation_weights)
                    
                # Explicitly free memory after evaluation
                del full_video
                torch.cuda.empty_cache()
            else:
                reward = 0.0
            
            print('Reward is:',reward)
            restore_wan_model(model)

            # Backpropagate the reward
            if selected_node.level == timesteps:
                current = selected_node
            else:
                current = selected_child
            while current:
                current.visits += 1
                current.value += reward
                current = current.parent
                
            if debug:
                print(f"[DEBUG] Simulation {sim+1}/{num_simulations}: Reward = {reward:.4f}")
    
    # Find best candidate (rank 0 only)
    final_state = None
    if rank == 0:
        # Collect all leaf nodes
        def collect_leaves(node):
            if not node.children:
                return [node]
            leaves = []
            for child in node.children:
                leaves.extend(collect_leaves(child))
            return leaves
        
        leaves = collect_leaves(root)
        if debug:
            print(f"[DEBUG] Collected {len(leaves)} leaf nodes")
        
        if leaves:
            # Find deepest leaves
            max_level = max(leaf.level for leaf in leaves)
            final_candidates = [leaf for leaf in leaves if leaf.level == max_level]
            
            # Select best candidate based on average reward
            best_candidate = max(final_candidates, key=lambda n: n.value / n.visits if n.visits > 0 else -float("inf"))

            if best_candidate.level < timesteps:
                print(f"[DEBUG] Best candidate at level {best_candidate.level} is not at full depth")
                print(f"[DEBUG] Best candidate is {best_candidate.video_id}")

            if debug:
                avg_reward = best_candidate.value / best_candidate.visits if best_candidate.visits > 0 else 0.0
                print(f"[DEBUG] Best candidate at level {best_candidate.level} with avg reward {avg_reward:.4f}")
            
            # Get the complete video for the best path
            path_segments = []
            current = best_candidate
            while current.parent:
                if current.video is not None:
                    # Move video to GPU for concatenation
                    path_segments.append(current.video.to(device=torch.device('cuda')))
                current = current.parent
            path_segments.reverse()
            
            if path_segments:
                # Resize tensors to match before concatenation
                # path_segments = resize_tensors_to_match(path_segments)
                final_state = torch.cat(path_segments, dim=1)
                if debug:
                    print(f"[DEBUG] Final video has {final_state.shape[1]} frames")
                    
                # Move final state to CPU
                final_state_cpu = final_state.clone().detach().cpu()
                
                # Clean up GPU memory
                del final_state
                del path_segments
                torch.cuda.empty_cache()
        
        if final_state_cpu is None:
            if debug:
                print("[DEBUG] No valid final state found")
            return None
    else:
        final_state_cpu = None
    
    # Broadcast final result to all processes
    if is_distributed:
        # First broadcast if a valid result exists
        has_result_val = [1 if final_state_cpu is not None else 0]
        if rank != 0:
            has_result_val = [0]
        dist.broadcast_object_list(has_result_val, src=0)
        has_result_val = has_result_val[0]
        
        if has_result_val == 1:
            if rank == 0:
                # Broadcast dimensions as a list instead of tensor
                dims_list = [list(final_state_cpu.shape)]
            else:
                # Empty list on other ranks
                dims_list = [None]
            
            # Broadcast dimensions using object_list
            dist.broadcast_object_list(dims_list, src=0)
            shape_dims = dims_list[0]
            
            # Create tensor on non-root processes
            if rank != 0:
                final_state_cpu = torch.zeros(shape_dims, dtype=torch.float32)
            
            # For large tensor data, we need a different approach since broadcast_object_list
            # may not handle large tensors well. We'll send the tensor in chunks.
            if rank == 0:
                # Serialize tensor to bytes
                buffer = final_state_cpu.cpu().numpy().tobytes()
                # Broadcast buffer size first
                buffer_size = [len(buffer)]
            else:
                buffer_size = [0]
            
            # Broadcast buffer size
            dist.broadcast_object_list(buffer_size, src=0)
            
            # Now handle the actual tensor data
            # In a real implementation, you might want to split this into chunks
            # For simplicity, we're using object broadcast which is safer but less efficient for large data
            if rank == 0:
                tensor_data = [final_state_cpu.cpu().numpy()]
            else:
                tensor_data = [None]
                
            dist.broadcast_object_list(tensor_data, src=0)
            
            if rank != 0:
                # Convert numpy array back to tensor
                final_state_cpu = torch.tensor(tensor_data[0], dtype=torch.float32)
    if rank == 0:
        visualize_mcts_tree(root, output_file=f"{output_path}/mcts_tree")
    return final_state_cpu

# Helper function to extract the last N frames from a video tensor
def extract_last_n_frames(video_tensor, n):
    """
    Extract the last N frames from a video tensor.
    
    Args:
        video_tensor (torch.Tensor): Video tensor with shape (C, T, H, W)
        n (int): Number of frames to extract
        
    Returns:
        list: List of PIL Images of the last N frames
    """
    # Get the total number of frames
    total_frames = video_tensor.shape[1]
    
    # Determine how many frames to extract (minimum of n and total frames)
    num_frames = min(n, total_frames)
    
    # Extract the last n frames as tensors
    frame_tensors = [video_tensor[:, total_frames - i - 1, :, :] for i in range(num_frames)]
    
    # Convert tensors to PIL Images
    frame_images = [tensor_to_pil(tensor) for tensor in frame_tensors]
    
    # Return in chronological order (oldest first)
    return frame_images[::-1]

# Helper function to uniformly sample N frames from a video tensor
def sample_uniform_frames(video_tensor, n):
    """
    Uniformly sample N frames from a video tensor.
    
    Args:
        video_tensor (torch.Tensor): Video tensor with shape (C, T, H, W)
        n (int): Number of frames to sample
        
    Returns:
        list: List of PIL Images of the sampled frames
    """
    # Get the total number of frames
    total_frames = video_tensor.shape[1]
    
    # Determine how many frames to extract (minimum of n and total frames)
    num_frames = min(n, total_frames)
    
    if num_frames == 1:
        # If only one frame is requested, return the last frame
        return [tensor_to_pil(video_tensor[:, -1, :, :])]
    
    if num_frames == total_frames:
        # If all frames are requested, just return all frames
        return [tensor_to_pil(video_tensor[:, i, :, :]) for i in range(total_frames)]
    
    # Calculate indices for uniform sampling
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    
    # Extract the frames at the calculated indices
    frame_tensors = [video_tensor[:, i, :, :] for i in indices]
    
    # Convert tensors to PIL Images
    frame_images = [tensor_to_pil(tensor) for tensor in frame_tensors]
    
    return frame_images
