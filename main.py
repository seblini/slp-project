import sys
from pathlib import Path

FAIRSEQ_PATH = Path(__file__).parent / "../av_hubert/fairseq"
sys.path.insert(0, str(FAIRSEQ_PATH))

AV_HUBERT_PATH = Path(__file__).parent / "../av_hubert/avhubert"
sys.path.insert(0, str(AV_HUBERT_PATH))

import fairseq # type: ignore
import hubert_pretraining, hubert # type: ignore

ckpt_path = "checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
