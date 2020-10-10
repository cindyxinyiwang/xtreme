import numpy as np
import torch


def switch_out(tokens, mask, tau, unk_token_id, pad_token_id, cls_token_id, sep_token_id, vocab_size):
    # first sample the number of words to corrupt
    max_len = tokens.size(1)

    pad_mask = (tokens == pad_token_id)
    cls_mask = (tokens == cls_token_id)
    sep_mask = (tokens == sep_token_id)
    sample_mask = ~((~pad_mask) & (~cls_mask) & (~sep_mask))

    logits = torch.arange(max_len).float().to(tokens.device)
    #mask = []
    #for i in lengths.tolist():
    #    mask.append([0 for _ in range(i)] + [1 for _ in range(max_len-i)])
    #mask = torch.LongTensor(mask).bool()
    lengths = mask.long().sum(dim=-1)
    # 1 for padding, 0 for tokens
    mask = (1-mask).bool()
    logits = logits.mul_(-1).unsqueeze(0).expand_as(tokens).contiguous().masked_fill_(mask, -float('inf'))
    probs = torch.softmax(logits.mul_(tau), dim=-1)
    num_words = torch.distributions.Categorical(probs).sample().float()
    lengths = lengths.float()

    # sample the indices to corrupt
    corrupt_pos = num_words.div_(lengths).unsqueeze(1).expand_as(tokens).contiguous().masked_fill_(sample_mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).byte().bool()
    total_words = int(corrupt_pos.sum())
    if total_words == 0:
        return tokens
    # sample the corrupts
    corrupt_val = torch.LongTensor(total_words).to(tokens.device)
    corrupts = torch.zeros_like(tokens).long().to(tokens.device)
    corrupts = corrupts.masked_scatter_(corrupt_pos, corrupt_val)
    sampled_tokens = tokens.add(corrupts).remainder_(vocab_size).masked_fill_(pad_mask, pad_token_id)
    return sampled_tokens



