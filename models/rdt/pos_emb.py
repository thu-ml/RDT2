from collections import OrderedDict

import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.
    
    Paper:
    https://arxiv.org/abs/1706.03762
    
    Source:
    https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
    
    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | list): a list of positions to be encoded, size (M,).
    Returns:
        out (ndarray): resulting positional embedding, size (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid_sizes):
    """
    Get ND positional embedding from grid sizes.
    All dimensions are summed up for factorization.
    
    Paper:
    https://arxiv.org/abs/2307.06304
    
    Args:
        embed_dim (int): output dimension for each position.
        grid_sizes (tuple): grids sizes in each dimension, length = K.
            If some grid size is lower than 1, we do not add any positional embedding.
    Returns:
        out (ndarray): resulting positional embedding, size (grid_sizes[0], ..., grid_sizes[K-1], D).
    """
    # We sum up all dimensions for factorization
    emb = np.zeros(grid_sizes + (embed_dim,))
    for size_idx, grid_size in enumerate(grid_sizes):
        # For grid size of 1, we do not need to add any positional embedding
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [embed_dim]
        posemb_shape[size_idx] = -1
        emb += get_1d_sincos_pos_embed_from_grid(embed_dim, pos).reshape(posemb_shape)
    return emb


def get_multimodal_pos_embed(embed_dim: int, mm_lens: OrderedDict):
    """
    Generate position embeddings for multimodal inputs. 
    
    Args:
        mm_lens (OrderedDict): an OrderedDict containing 
            (modality name, modality token length) pairs.
            For `"image"` modality, the value can be a multi-dimensional tuple.
            If the length < 0, it means there is no position embedding for the modality or grid.
    Returns:
        out (ndarray): positional embeddings for multimodal inputs, size (seq_len, embed_dim).
    """
    # Get total length
    tot_len = 0
    for modality, cond_len in mm_lens.items():
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            tot_len += np.prod([abs(x) for x in cond_len])
        else:
            tot_len += abs(cond_len)
    
    num_modalities = len(mm_lens)
    if num_modalities > 1:
        # Embed modality information
        modality_pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, np.arange(num_modalities))
        
    # Get embeddings for positions inside each modality
    pos_emb = np.zeros((tot_len, embed_dim))
    start_pos = 0
    for idx, (modality, cond_len) in enumerate(mm_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            pos_embed_i_ = get_nd_sincos_pos_embed_from_grid(
                embed_dim, embed_grid_sizes)
            pos_embed_i = np.zeros(all_grid_sizes + (embed_dim,))
            pos_embed_i += pos_embed_i_
            pos_embed_i = pos_embed_i.reshape((-1, embed_dim))
        else:
            pos_embed_i_ = get_1d_sincos_pos_embed_from_grid(
                embed_dim, np.arange(cond_len)) if cond_len > 1 else 0
            pos_embed_i = np.zeros((abs(cond_len), embed_dim))
            pos_embed_i += pos_embed_i_
        
        if num_modalities > 1:
            pos_embed_i += modality_pos_embed[idx]
        # Aggregate the positional embeddings
        pos_emb[start_pos:start_pos + len(pos_embed_i)] = pos_embed_i
        start_pos += len(pos_embed_i)
    
    return pos_emb
