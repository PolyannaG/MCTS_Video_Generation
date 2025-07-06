import av
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification

# Load the model and processor
model_name = "TIGER-Lab/VideoScore"
processor = AutoProcessor.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = Idefics2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

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

def evaluate_video(video_path, text_prompts, frame_num):
    if device.type=='cuda':
        model.to(device)
    frames = extract_all_frames(video_path)
    score = run_window_evaluation(frames, text_prompts, frame_num)
    if device.type=='cuda':
        model.to("cpu")
    return score

# # Example
# video_path = "examples/0_0_1.mp4"
# text_prompt = "A purple wall with a brown baseboard stretches behind a pink couch, round wood end table with a yellow shaded purple lamp and green box atop it. Tom, the blue-gray cat, is reclining with his right leg over his left, holding yellow darts in his left hand while he throws them casually out of the left side of the frame, one at a time."
# scores = evaluate_video(video_path, text_prompt)
# print("Averaged Aspect Scores:", scores)



# [2.898, 2.398, 2.68, 2.891, 2.148]
