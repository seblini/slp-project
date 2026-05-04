import json
b = json.load(open('runs/student_v1/eval_full.json'))
v = json.load(open('runs/student_viseme/eval_full.json'))

print(f'{"Metric":<25} {"baseline":>12} {"viseme":>12} {"delta":>10}')
print('-' * 62)

for path, label in [
    (('distribution_match', 'kl'), 'Pure KL'),
    (('distribution_match', 'cross_entropy'), 'Cross-entropy'),
    (('distribution_match', 'top1_agreement'), 'Top-1 agreement'),
    (('distribution_match', 'top5_agreement'), 'Top-5 agreement'),
    (('decoding', 'word_presence_acc'), 'Word presence'),
    (('decoding', 'wer_vs_teacher'), 'WER vs teacher'),
]:
    bv, vv = b, v
    for k in path:
        bv = bv[k]
        vv = vv[k]
    delta = vv - bv
    print(f'{label:<25} {bv:>12.4f} {vv:>12.4f} {delta:>+10.4f}')
