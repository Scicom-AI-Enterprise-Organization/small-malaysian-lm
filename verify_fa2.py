import random
import numpy as np
import flash_attn
import torch

def generate_list_sum_n(n, length=5, min_val=5):

    numbers = [min_val] * length
    remaining = n - min_val * length

    for _ in range(remaining):
        numbers[random.randint(0, length - 1)] += 1

    random.shuffle(numbers)
    return numbers

sequence_length = 4096
query_lens = np.array(generate_list_sum_n(sequence_length, length=20, min_val=10), dtype=np.int64)

q = torch.randn(1, sequence_length, 128, 128, dtype = torch.bfloat16).cuda()
k = torch.randn(1, sequence_length, 128, 128, dtype = torch.bfloat16).cuda()
v = torch.randn(1, sequence_length, 128, 128, dtype = torch.bfloat16).cuda()

cumsum = [0] + np.cumsum(query_lens).tolist()
max_cumsum = int(np.max(cumsum))
cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32).cuda()
max_seqlen_q = np.max(query_lens)

out_flash2 = flash_attn.flash_attn_varlen_func(
    q = q[0],
    k = k[0],
    v = v[0],
    cu_seqlens_q = cu_seq_lens_q,
    cu_seqlens_k = cu_seq_lens_q,
    max_seqlen_q = max_seqlen_q,
    max_seqlen_k = max_seqlen_q,
    causal = True
)