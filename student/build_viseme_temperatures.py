"""
Build a per-token temperature table based on viseme ambiguity.

For each token in the teacher's vocabulary, we:
1. Convert the BPE piece to its phoneme sequence (via g2p_en)
2. Map phonemes to viseme classes (Jeffers-Barley scheme)
3. Compute an ambiguity score based on visually-confusable phonemes
4. Convert ambiguity to temperature: T_min for unambiguous, T_max for ambiguous

The output is a (vocab_size,) array saved to disk.
"""
import sys
import re
import argparse
from pathlib import Path
import numpy as np

AVHUBERT_PARENT = (Path(__file__).parent / "../../av_hubert").resolve()
FAIRSEQ_PATH = (Path(__file__).parent / "../../av_hubert/fairseq").resolve()
sys.path.insert(0, str(AVHUBERT_PARENT))
sys.path.insert(0, str(FAIRSEQ_PATH))

import fairseq  # noqa
from avhubert import hubert_pretraining  # noqa
from avhubert import hubert  # noqa
from avhubert import hubert_asr  # noqa


PHONEME_TO_VISEME = {
    # V1 — Bilabial: /p/, /b/, /m/ are visually identical (lips closed)
    'P': 'V1', 'B': 'V1', 'M': 'V1',
    # V2 — Labio-dental: /f/, /v/ (teeth on lower lip)
    'F': 'V2', 'V': 'V2',
    # V3 — Dental: /th/ (tongue between teeth)
    'TH': 'V3', 'DH': 'V3',
    # V4 — Alveolar: /t/, /d/, /s/, /z/, /n/, /l/ (tongue behind teeth, mostly invisible)
    'T': 'V4', 'D': 'V4', 'S': 'V4', 'Z': 'V4', 'N': 'V4', 'L': 'V4',
    # V5 — Post-alveolar: /sh/, /zh/, /ch/, /j/
    'SH': 'V5', 'ZH': 'V5', 'CH': 'V5', 'JH': 'V5',
    # V6 — Velar: /k/, /g/, /ng/ (tongue at back, completely invisible)
    'K': 'V6', 'G': 'V6', 'NG': 'V6',
    # V7 — Glottal
    'HH': 'V7',
    # V8 — Approximants
    'R': 'V8', 'Y': 'V8', 'W': 'V8',
    # V9 — Open vowels
    'AA': 'V9', 'AO': 'V9', 'AE': 'V9', 'AW': 'V9', 'AY': 'V9',
    # V10 — Mid vowels
    'AH': 'V10', 'AX': 'V10', 'EH': 'V10', 'ER': 'V10',
    # V11 — Close vowels
    'IH': 'V11', 'IY': 'V11', 'EY': 'V11',
    # V12 — Round vowels
    'UH': 'V12', 'UW': 'V12', 'OW': 'V12', 'OY': 'V12',
}

# Visemes where multiple phonemes share the class — these are visually ambiguous
HIGH_AMBIGUITY_VISEMES = {'V1', 'V2'}


def strip_stress(phoneme):
    return re.sub(r'\d+$', '', phoneme)


def get_phonemes(text, g2p):
    word = text.replace('▁', '').strip()
    if not word:
        return []
    if not any(c.isalpha() for c in word):
        return []
    # Skip special tokens that look like <unk>, <pad>, <s>, </s>
    if word.startswith('<') or word.endswith('>'):
        return []
    phonemes = g2p(word)
    cleaned = []
    for p in phonemes:
        p = strip_stress(p)
        if p in PHONEME_TO_VISEME:
            cleaned.append(p)
    return cleaned


def viseme_ambiguity(phonemes):
    """
    Score 0-1: fraction of phonemes belonging to highly-ambiguous viseme classes.
    Returns 0.0 for tokens with no phonemes (special tokens, punctuation).
    """
    if not phonemes:
        return 0.0
    visemes = [PHONEME_TO_VISEME[p] for p in phonemes]
    n_ambig = sum(1 for v in visemes if v in HIGH_AMBIGUITY_VISEMES)
    return n_ambig / len(visemes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher_ckpt', default='data/checkpoints/checkpoint.pt')
    ap.add_argument('--output', default='data/token_temperatures.npy')
    ap.add_argument('--t_min', type=float, default=1.0,
                    help='Temperature for fully unambiguous tokens')
    ap.add_argument('--t_max', type=float, default=4.0,
                    help='Temperature for fully ambiguous tokens')
    ap.add_argument('--default_t', type=float, default=2.0,
                    help='Temperature for special / non-alphabetic tokens')
    ap.add_argument('--print_examples', type=int, default=20)
    args = ap.parse_args()

    print(f"Loading teacher dict from {args.teacher_ckpt}")
    _, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [args.teacher_ckpt])
    d = task.target_dictionary
    vocab_size = len(d)
    print(f"Vocab size: {vocab_size}")

    from g2p_en import G2p
    g2p = G2p()

    temperatures = np.full(vocab_size, args.default_t, dtype=np.float32)
    ambiguities = np.zeros(vocab_size, dtype=np.float32)

    n_with_phonemes = 0
    examples = []

    for token_id in range(vocab_size):
        token_str = d.string([token_id])
        phonemes = get_phonemes(token_str, g2p)
        if not phonemes:
            continue

        n_with_phonemes += 1
        ambig = viseme_ambiguity(phonemes)
        ambiguities[token_id] = ambig
        T = args.t_min + (args.t_max - args.t_min) * ambig
        temperatures[token_id] = T

        if len(examples) < args.print_examples * 3:
            examples.append({
                'token_id': token_id,
                'token_str': token_str.strip(),
                'phonemes': phonemes,
                'ambiguity': ambig,
                'temperature': T,
            })

    np.save(args.output, temperatures)
    print(f"\nSaved temperatures to {args.output}")
    print(f"  tokens with phonemes: {n_with_phonemes}/{vocab_size}")
    print(f"  default temperature for {vocab_size - n_with_phonemes} non-phoneme tokens: "
          f"{args.default_t}")
    print(f"  temperature range: [{temperatures.min():.2f}, {temperatures.max():.2f}]")
    print(f"  mean temperature: {temperatures.mean():.2f}")
    print(f"  ambiguity histogram:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((ambiguities >= lo) & (ambiguities < hi) & (ambiguities > 0)).sum()
        print(f"    [{lo:.1f}, {hi:.1f}): {n}")

    print(f"\nExamples (sorted by ambiguity):")
    examples.sort(key=lambda x: x['ambiguity'])
    n_show = min(args.print_examples, len(examples) // 2)
    print("  Most unambiguous:")
    for ex in examples[:n_show]:
        print(f"    [{ex['token_id']:4d}] '{ex['token_str']:<15}' "
              f"phonemes={ex['phonemes']!r:<40} "
              f"ambig={ex['ambiguity']:.2f} T={ex['temperature']:.2f}")
    print("  Most ambiguous:")
    for ex in examples[-n_show:]:
        print(f"    [{ex['token_id']:4d}] '{ex['token_str']:<15}' "
              f"phonemes={ex['phonemes']!r:<40} "
              f"ambig={ex['ambiguity']:.2f} T={ex['temperature']:.2f}")


if __name__ == '__main__':
    main()
