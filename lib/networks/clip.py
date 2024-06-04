import clip

import torch
import torch.nn.functional as F


def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encoded_text(clip_model, text):
    text_token = tokenize(text)
    enc_text = clip_model.encode_text(text_token).float()
    return enc_text

def encoded_text_normalized(clip_model, text):
    enc_text = encoded_text(clip_model, text)
    normalized_vector = F.normalize(enc_text, p=2, dim=1)
    return normalized_vector

def tokenize(raw_text, device="cuda"):
    max_text_len = 20

    default_context_length = 77
    context_length = max_text_len + 2 # start_token + 20 + end_token
    assert context_length < default_context_length
    texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
    # print('texts', texts.shape)
    zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
    texts = torch.cat([texts, zero_pad], dim=1)
    return texts