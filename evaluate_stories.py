import av
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification
import json
import os
import argparse

# Load the model and processor
model_name = "TIGER-Lab/VideoScore"
processor = AutoProcessor.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = Idefics2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality
(2) temporal consistency
(3) dynamic degree
(4) text-to-video alignment
(5) factual consistency

Each dimension: float number from 1.0 to 4.0
Example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

def extract_all_frames(video_path):
    container = av.open(video_path)
    frames = [Image.fromarray(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
    return frames

def run_window_evaluation(frames, text_prompts, frame_num):
    if len(text_prompts) == 1:
        # only one prompt, so we only need to evaluate the video once
        segment = frames
        prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=text_prompts[0]) + "<image> " * len(segment)
        inputs = processor(text=prompt, images=segment, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        scores = logits[0].cpu().tolist()
        return [round(s, 3) for s in scores]
    all_scores = []
    index = 0
    for index in range(0, len(text_prompts)-1):
        segment = frames[index*frame_num:(index+2)*frame_num]
        prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=text_prompts[index]+" "+text_prompts[index+1]) + "<image> " * len(segment)
        inputs = processor(text=prompt, images=segment, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        scores = logits[0].cpu().tolist()
        all_scores.append(scores)

    if not all_scores:
        print("Not enough frames to evaluate a full window.")
        return [None] * 5

    avg_scores = np.mean(all_scores, axis=0)
    return [round(s, 3) for s in avg_scores]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories_dir", type=str, required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    stories_dir = args.stories_dir
    video_dir = os.path.join(stories_dir, "videos")
    video_files = os.listdir(video_dir)

    # create output dir
    output_dir = os.path.join(stories_dir, "scores")
    os.makedirs(output_dir, exist_ok=True)

    # sort video files video_01.mp4
    if "_" in video_files[0]:
        video_files.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
    else:
        video_files.sort(key=lambda x: int(x.split(".")[0]))


    prompts_file = os.path.join(stories_dir, "prompts.json")
    # read prompts
    with open(prompts_file, "r") as f:
        prompts = json.load(f)
    assert len(prompts) == len(video_files) , "The number of prompts and videos are not the same"

    # read videos
    for i, video_file in enumerate(video_files):
        try:
            print(f"Processing {video_file}")
            video_path = os.path.join(video_dir, video_file)
            if type(prompts[i]['id']) == int:
                assert int(video_file.split(".")[0]) == prompts[i]["id"], "The video file and prompt file are not the same"
            else:
                assert video_file.split(".")[0] == prompts[i]["id"].split(".")[0], "The video file and prompt file are not the same"

            video_prompt = prompts[i]["prompts"]
            frames = extract_all_frames(video_path)
            frame_num = int(len(frames) / len(video_prompt))
            print(f"Frame number: {frame_num}")
            scores = run_window_evaluation(frames, video_prompt, frame_num)
            print(scores)

            # save scores
            output_file = os.path.join(output_dir, f"{video_file.split('.')[0]}.json")
            with open(output_file, "w") as f:
                    json.dump(scores, f)    
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue

if __name__ == "__main__":
    main()