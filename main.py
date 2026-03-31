import sys
from pathlib import Path

FAIRSEQ_PATH = Path(__file__).parent / "../av_hubert/fairseq"
sys.path.insert(0, str(FAIRSEQ_PATH))

AV_HUBERT_PATH = Path(__file__).parent / "../av_hubert/avhubert"
sys.path.insert(0, str(AV_HUBERT_PATH))

import fairseq # type: ignore

import urllib.request
import os

from mp import process_video, save_crops_as_video, plot_crops

from model_utils import prep_inference, run_inference_and_extract_soft_targets, crops_to_tensor

import logging
import warnings

from student_model import StudentLipReader, DistillationTrainer

import torch

# Remove noisy warnings
warnings.filterwarnings("ignore")

# Remove noisy logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("hubert_pretraining").setLevel(logging.WARNING)
logging.getLogger("hubert").setLevel(logging.WARNING)
logging.getLogger("hubert_dataset").setLevel(logging.WARNING)

# Load AV Hubert model
ckpt_path = "checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0].eval()

# Download mediapipe model
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
if not os.path.exists("face_landmarker.task"):
    urllib.request.urlretrieve(url, "face_landmarker.task")

# Process video using mediapipe
video_path = "AFTERNOON.mp4"
crops, fps = process_video(video_path)

plot_crops(crops)

save_crops_as_video(crops, "AFTERNOON-roi.mp4")

# Prepare inference
itr, generator, hypo_token_decoder  = prep_inference(os.path.abspath("AFTERNOON-roi.mp4"), model, cfg, task)

# Run inference and extract soft targets
soft_targets = run_inference_and_extract_soft_targets(model, itr)

VOCAB_SIZE = 1000
PAD_IDX = 1
BOS_IDX = 0
BATCH_SIZE = 1
NUM_FRAMES = 29
SEQ_LEN = 5

# Initialize student model
student = StudentLipReader(
    vocab_size=VOCAB_SIZE,
    embed_dim=256,
    encoder_layers=4,
    decoder_layers=4,
    n_heads=4,
    ff_dim=512,
    pad_idx=PAD_IDX,
    freeze_early_resnet=True,
)
print()
student.print_parameter_breakdown()

# Prepare training data
video_frames = crops_to_tensor(crops)
prev_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
prev_tokens[:, 0] = BOS_IDX
hard_targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
teacher_soft_targets = torch.softmax(torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), dim=-1)

# Run training step
trainer = DistillationTrainer(student, temperature=2.0, alpha=0.7)
losses = trainer.train_step(video_frames, prev_tokens, teacher_soft_targets, hard_targets)

print(f"\nCombined loss: {losses['loss']:.4f}")
print(f"Soft loss:     {losses['soft_loss']:.4f}")
print(f"Hard loss:     {losses['hard_loss']:.4f}")
