# evaluate_student.py
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

AVHUBERT_PARENT = (Path(__file__).parent / "../../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(FAIRSEQ_PATH))

import fairseq  # noqa
from avhubert import hubert_pretraining, hubert, hubert_asr  # noqa

from student_model import VideoStudent
from student_dataset import (
    LRWDistillationDataset, collate_fn, split_clip_ids_by_lrw_split,
)


def shift_right_for_teacher_forcing(tokens, bos_id, pad_id):
    B, T = tokens.shape
    bos = torch.full((B, 1), bos_id, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([bos, tokens[:, :-1]], dim=1)


def decode_to_text(token_ids, dictionary):
    text = dictionary.string(token_ids)
    text = text.replace('▁', ' ').strip()
    text = ' '.join(text.split())
    return text


def word_presence(pred_text, target_word):
    return target_word.upper() in pred_text.upper().split()


def compute_wer(reference, hypothesis):
    ref = reference.split()
    hyp = hypothesis.split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    R, H = len(ref), len(hyp)
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[R][H] / R


@torch.no_grad()
def eval_distribution_match(model, loader, special, device, temperature=2.0):
    """
    Teacher-forced pass: compute distribution-level metrics aligned with training.
    
    Returns:
        cross_entropy: -E[t_soft * log(s_soft)] (what train val_kl reports)
        teacher_entropy: -E[t_soft * log(t_soft)] (constant w.r.t. student)
        kl: cross_entropy - teacher_entropy (pure KL divergence)
        top1_agreement: P(argmax(student) == argmax(teacher))
        top5_agreement: P(argmax(teacher) ∈ top5(student))
    """
    model.eval()
    total_ce = 0.0
    total_h = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0
    
    for batch in tqdm(loader, desc='distribution match'):
        video = batch['video'].to(device, non_blocking=True)
        video_mask = batch['video_mask'].to(device, non_blocking=True)
        teacher_logits = batch['teacher_logits'].to(device, non_blocking=True)
        teacher_tokens = batch['teacher_tokens'].to(device, non_blocking=True)
        decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
        
        prev_tokens = shift_right_for_teacher_forcing(
            teacher_tokens, special['bos_id'], special['pad_id'])
        
        student_logits = model(video, video_mask, prev_tokens,
                               decoder_mask=decoder_mask)
        
        # Distribution metrics at temperature T
        s_log = F.log_softmax(student_logits / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
        
        ce_per_tok = -(t_soft * s_log).sum(dim=-1)
        h_per_tok = -(t_soft * t_log).sum(dim=-1)
        
        valid = (~decoder_mask).float()
        n = valid.sum().item()
        
        total_ce += (ce_per_tok * valid).sum().item()
        total_h += (h_per_tok * valid).sum().item()
        
        # Argmax agreement (uses raw logits, not temperature-scaled)
        s_top1 = student_logits.argmax(dim=-1)  # (B, T_d)
        t_top1 = teacher_logits.argmax(dim=-1)
        agree_top1 = (s_top1 == t_top1).float() * valid
        total_top1 += agree_top1.sum().item()
        
        # Top-5 agreement: is teacher's top in student's top 5?
        s_top5 = student_logits.topk(k=5, dim=-1).indices  # (B, T_d, 5)
        t_top1_exp = t_top1.unsqueeze(-1)
        in_top5 = (s_top5 == t_top1_exp).any(dim=-1).float() * valid
        total_top5 += in_top5.sum().item()
        
        total_tokens += n
    
    n = max(total_tokens, 1)
    return {
        'cross_entropy': total_ce / n,
        'teacher_entropy': total_h / n,
        'kl': (total_ce - total_h) / n,
        'top1_agreement': total_top1 / n,
        'top5_agreement': total_top5 / n,
        'n_tokens': total_tokens,
    }


@torch.no_grad()
def eval_decoding(model, loader, dictionary, device, max_decode_len=20,
                  beam_size=1, length_penalty=1.0):
    """
    Decode-and-compare pass: word presence accuracy and WER vs teacher.
    """
    model.eval()
    n_total = 0
    n_word_correct = 0
    wer_sum = 0.0
    wer_count = 0
    per_word_total = {}
    per_word_correct = {}
    examples = []
    
    for batch in tqdm(loader, desc=f'decoding (beam={beam_size})'):
        video = batch['video'].to(device, non_blocking=True)
        video_mask = batch['video_mask'].to(device, non_blocking=True)
        clip_ids = batch['clip_ids']
        target_words = batch['target_words']
        teacher_tokens = batch['teacher_tokens']
        
        if beam_size > 1:
            student_tokens = model.beam_search_decode(
                video, video_mask, beam_size=beam_size,
                max_len=max_decode_len, length_penalty=length_penalty,
            )
        else:
            student_tokens = model.greedy_decode(
                video, video_mask, max_len=max_decode_len)
        
        for i, cid in enumerate(clip_ids):
            teacher_len = batch['decoder_lens'][i].item()
            t_toks = teacher_tokens[i, :teacher_len].tolist()
            teacher_text = decode_to_text(t_toks, dictionary)
            
            s_toks = student_tokens[i].tolist()
            try:
                eos_idx = s_toks.index(dictionary.eos())
                s_toks = s_toks[:eos_idx + 1]
            except ValueError:
                pass
            student_text = decode_to_text(s_toks, dictionary)
            
            target_word = target_words[i]
            n_total += 1
            is_correct = word_presence(student_text, target_word)
            if is_correct:
                n_word_correct += 1
            
            per_word_total[target_word] = per_word_total.get(target_word, 0) + 1
            if is_correct:
                per_word_correct[target_word] = per_word_correct.get(target_word, 0) + 1
            
            if teacher_text.strip():
                wer_sum += compute_wer(teacher_text, student_text)
                wer_count += 1
            
            if len(examples) < 20:
                examples.append({
                    'clip_id': cid,
                    'target': target_word,
                    'teacher': teacher_text,
                    'student': student_text,
                    'correct': is_correct,
                })
    
    return {
        'word_presence_acc': n_word_correct / n_total if n_total else 0.0,
        'wer_vs_teacher': wer_sum / wer_count if wer_count else 0.0,
        'n_clips': n_total,
        'examples': examples,
        'per_word_total': per_word_total,
        'per_word_correct': per_word_correct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--videos', required=True)
    ap.add_argument('--logits', required=True)
    ap.add_argument('--teacher_ckpt', default='data/checkpoints/checkpoint.pt')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_decode_len', type=int, default=20)
    ap.add_argument('--beam_size', type=int, default=1)
    ap.add_argument('--length_penalty', type=float, default=1.0)
    ap.add_argument('--temperature', type=float, default=2.0,
                    help='Temperature for distribution-match metrics')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output', default=None)
    ap.add_argument('--skip_decode', action='store_true',
                    help='Skip greedy/beam decoding pass (only distribution metrics)')
    args = ap.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading teacher dict from {args.teacher_ckpt}")
    _, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.teacher_ckpt])
    d = task.target_dictionary
    
    special = {
        'pad_id': d.pad(), 'bos_id': d.bos(), 'eos_id': d.eos(),
    }
    
    print(f"Loading student from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    student = VideoStudent(
        vocab_size=len(d),
        pad_id=d.pad(), bos_id=d.bos(), eos_id=d.eos(),
    ).to(device).eval()
    student.load_state_dict(ckpt['model'])
    print(f"  loaded epoch {ckpt.get('epoch', '?')}, "
          f"train val_kl={ckpt.get('val_kl', '?')}")
    
    full = LRWDistillationDataset(args.videos, args.logits)
    splits = split_clip_ids_by_lrw_split(full.clip_ids)
    split_ids = splits[args.split]
    if args.limit:
        split_ids = split_ids[:args.limit]
    
    eval_set = LRWDistillationDataset(args.videos, args.logits, clip_ids=split_ids)
    loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"\nEvaluating on {len(eval_set)} {args.split} clips...")
    
    # --- Distribution match pass (teacher-forced) ---
    print("\n[1/2] Distribution-match metrics (teacher-forced)")
    dist = eval_distribution_match(student, loader, special, device,
                                    temperature=args.temperature)
    
    print(f"\n  Cross-entropy (T={args.temperature}): {dist['cross_entropy']:.4f}")
    print(f"  Teacher entropy:                 {dist['teacher_entropy']:.4f}")
    print(f"  Pure KL:                         {dist['kl']:.4f}")
    print(f"  Top-1 agreement:                 {dist['top1_agreement']*100:.2f}%")
    print(f"  Top-5 agreement:                 {dist['top5_agreement']*100:.2f}%")
    print(f"  Tokens evaluated:                {dist['n_tokens']}")
    
    decode = None
    if not args.skip_decode:
        # --- Decode-and-compare pass ---
        print(f"\n[2/2] Decode-and-compare metrics (beam_size={args.beam_size})")
        decode = eval_decoding(student, loader, d, device,
                               max_decode_len=args.max_decode_len,
                               beam_size=args.beam_size,
                               length_penalty=args.length_penalty)
        
        print(f"\n  Word presence accuracy: {decode['word_presence_acc']*100:.2f}%")
        print(f"  WER vs teacher:         {decode['wer_vs_teacher']*100:.2f}%")
        print(f"  Clips evaluated:        {decode['n_clips']}")
        
        print(f"\n  Sample outputs:")
        print(f"  {'target':<12} {'?':<3} {'teacher':<40} {'student'}")
        print("  " + "-" * 100)
        for ex in decode['examples'][:10]:
            marker = '✓' if ex['correct'] else '✗'
            print(f"  {ex['target']:<12} {marker:<3} {ex['teacher']:<40} {ex['student']}")
    
    # --- Save ---
    if args.output:
        out = {
            'distribution_match': dist,
            'split': args.split,
            'ckpt_epoch': ckpt.get('epoch'),
            'beam_size': args.beam_size,
            'temperature': args.temperature,
        }
        if decode is not None:
            out['decoding'] = {
                'word_presence_acc': decode['word_presence_acc'],
                'wer_vs_teacher': decode['wer_vs_teacher'],
                'n_clips': decode['n_clips'],
                'per_word_acc': {
                    w: decode['per_word_correct'].get(w, 0) / t
                    for w, t in decode['per_word_total'].items()
                },
            }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == '__main__':
    main()
