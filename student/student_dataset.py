import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class LRWDistillationDataset(Dataset):
    """
    Loads paired (video ROIs, teacher logits, teacher tokens) from HDF5 files.
    
    Files should have matching keys (clip IDs).
    Video file: each key → (T, 96, 96) uint8 array
    Logits file: each key → group with 'logits' (T_dec, vocab) and 'tokens' (T_dec,)
    """
    
    def __init__(self, video_h5_path, logits_h5_path, clip_ids=None,
                 video_mean=0.421, video_std=0.165):
        self.video_path = video_h5_path
        self.logits_path = logits_h5_path
        self.video_mean = video_mean
        self.video_std = video_std
        
        # Determine clip IDs (intersection of both files)
        with h5py.File(video_h5_path, 'r') as v, h5py.File(logits_h5_path, 'r') as l:
            v_keys = set(v.keys())
            l_keys = set(l.keys())
            common = v_keys & l_keys
            print(f"  videos: {len(v_keys)}, logits: {len(l_keys)}, common: {len(common)}")
        
        if clip_ids is not None:
            # Filter to a specific subset (e.g., train/val/test split)
            common = common & set(clip_ids)
        
        self.clip_ids = sorted(common)
        
        # Lazy-open file handles per worker (HDF5 isn't picklable)
        self._video_h5 = None
        self._logits_h5 = None
    
    def _ensure_open(self):
        if self._video_h5 is None:
            self._video_h5 = h5py.File(self.video_path, 'r')
            self._logits_h5 = h5py.File(self.logits_path, 'r')
    
    def __len__(self):
        return len(self.clip_ids)
    
    def __getitem__(self, idx):
        self._ensure_open()
        cid = self.clip_ids[idx]
        
        # Video: (T, 96, 96) uint8 → (T, 1, 96, 96) float, normalized
        video = self._video_h5[cid][:]  # numpy
        video = torch.from_numpy(video).float() / 255.0
        video = (video - self.video_mean) / self.video_std
        video = video.unsqueeze(1)  # (T, 1, 96, 96)
        
        # Teacher outputs
        grp = self._logits_h5[cid]
        teacher_logits = torch.from_numpy(grp['logits'][:]).float()  # (T_dec, vocab)
        teacher_tokens = torch.from_numpy(grp['tokens'][:]).long()   # (T_dec,)
        
        # Target word (label) extracted from clip ID
        target_word = cid.split('_')[0]
        
        return {
            'clip_id': cid,
            'video': video,
            'teacher_logits': teacher_logits,
            'teacher_tokens': teacher_tokens,
            'target_word': target_word,
        }


def collate_fn(batch):
    """Pad variable-length clips and decoder sequences."""
    clip_ids = [b['clip_id'] for b in batch]
    target_words = [b['target_word'] for b in batch]
    
    videos = [b['video'] for b in batch]
    teacher_logits = [b['teacher_logits'] for b in batch]
    teacher_tokens = [b['teacher_tokens'] for b in batch]
    
    # Video lengths and padding
    v_lens = torch.tensor([v.shape[0] for v in videos])
    max_v = v_lens.max().item()
    B = len(batch)
    video_batch = torch.zeros(B, max_v, 1, 96, 96)
    video_mask = torch.ones(B, max_v, dtype=torch.bool)
    for i, v in enumerate(videos):
        T = v.shape[0]
        video_batch[i, :T] = v
        video_mask[i, :T] = False
    
    # Decoder sequence lengths and padding
    d_lens = torch.tensor([t.shape[0] for t in teacher_tokens])
    max_d = d_lens.max().item()
    vocab = teacher_logits[0].shape[-1]
    
    logits_batch = torch.zeros(B, max_d, vocab)
    tokens_batch = torch.zeros(B, max_d, dtype=torch.long)
    decoder_mask = torch.ones(B, max_d, dtype=torch.bool)
    for i, (lg, tk) in enumerate(zip(teacher_logits, teacher_tokens)):
        T_d = lg.shape[0]
        logits_batch[i, :T_d] = lg
        tokens_batch[i, :T_d] = tk
        decoder_mask[i, :T_d] = False
    
    return {
        'clip_ids': clip_ids,
        'target_words': target_words,
        'video': video_batch,           # (B, T_v, 1, 96, 96)
        'video_mask': video_mask,       # (B, T_v) - True = padded
        'video_lens': v_lens,           # (B,)
        'teacher_logits': logits_batch, # (B, T_d, vocab)
        'teacher_tokens': tokens_batch, # (B, T_d)
        'decoder_mask': decoder_mask,   # (B, T_d) - True = padded
        'decoder_lens': d_lens,         # (B,)
    }


def split_clip_ids_by_lrw_split(all_clip_ids):
    """LRW clip IDs encode the split: {WORD}_{split}_{filename}. Group accordingly."""
    splits = {'train': [], 'val': [], 'test': []}
    for cid in all_clip_ids:
        parts = cid.split('_')
        # parts = [WORD, split, ..., filename]
        split = parts[1]
        if split in splits:
            splits[split].append(cid)
    return splits


# ----- Quick smoke test -----

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', default='videos.h5')
    ap.add_argument('--logits', default='logits_full.h5')
    args = ap.parse_args()
    
    print("Loading dataset...")
    full = LRWDistillationDataset(args.videos, args.logits)
    print(f"Total clips: {len(full)}")
    
    # Look at one item
    item = full[0]
    print(f"\nFirst clip: {item['clip_id']}")
    print(f"  target word: {item['target_word']}")
    print(f"  video shape: {item['video'].shape}")
    print(f"  teacher_logits shape: {item['teacher_logits'].shape}")
    print(f"  teacher_tokens: {item['teacher_tokens'].tolist()}")
    
    # Split by LRW's predefined train/val/test
    splits = split_clip_ids_by_lrw_split(full.clip_ids)
    print(f"\nSplit sizes:")
    for split, ids in splits.items():
        print(f"  {split}: {len(ids)}")
    
    # Build a small dataloader and iterate one batch
    train_set = LRWDistillationDataset(args.videos, args.logits,
                                        clip_ids=splits['train'][:64])
    loader = DataLoader(train_set, batch_size=8, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  video: {batch['video'].shape}")
    print(f"  video_mask: {batch['video_mask'].shape}")
    print(f"  teacher_logits: {batch['teacher_logits'].shape}")
    print(f"  teacher_tokens: {batch['teacher_tokens'].shape}")
    print(f"  decoder_mask: {batch['decoder_mask'].shape}")
    print(f"  vocab size from logits: {batch['teacher_logits'].shape[-1]}")
