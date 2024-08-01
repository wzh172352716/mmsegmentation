import torch
import math

#checkpoint = dict(torch.load("/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b0_20220624-7e0fe6dd.pth"))
#checkpoint = dict(torch.load("/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b1_20220624-02e5a6a1.pth"))
#checkpoint = dict(torch.load("/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b2_20220624-66e8bf70.pth"))
#checkpoint = dict(torch.load("/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b3_20220624-13b1141c.pth"))
#checkpoint = dict(torch.load("/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b4_20220624-d588d980.pth"))
checkpoint = dict(torch.load("/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b5_20220624-658746d9.pth"))


if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

attn_keys = [k for k in checkpoint.keys() if "in_proj" in k]
attn_keys_out = [k for k in checkpoint.keys() if "out_proj" in k]

def mod(w, b, out_w, out_b, r=0.5, embed_dim = 32):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    dim_out, dim_in = w_q.shape
    num_heads = dim_out // embed_dim
    new_dim_out = int(dim_out * r)
    new_embed_dim = min(embed_dim, new_dim_out)
    new_num_heads = math.ceil(new_dim_out / new_embed_dim)
    new_dim_out = new_num_heads * new_embed_dim
    print("new_embed_dim: ", new_embed_dim)
    print("dim_out: ", dim_out)
    print("new_dim_out: ", new_dim_out)
    print("num_heads: ", num_heads)
    print("new_num_heads: ", new_num_heads)
    w = torch.vstack([w_q[0:new_dim_out, :], w_k[0:new_dim_out, :], w_v[0:new_dim_out, :]])
    b = torch.hstack([b_q[0:new_dim_out], b_k[0:new_dim_out], b_v[0:new_dim_out]])
    out_w = out_w[:, 0:new_dim_out]

    return w, b, out_w, out_b

def mod_small(w, b, out_w, out_b, r=0.5, embed_dim = 32):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    dim_out, dim_in = w_q.shape
    num_heads = dim_out // embed_dim
    new_dim_out = int(dim_out * r)
    new_embed_dim = min(embed_dim, new_dim_out)
    new_num_heads = new_dim_out // new_embed_dim
    new_dim_out = new_num_heads * new_embed_dim

    print("new_embed_dim: ", new_embed_dim)
    print("dim_out: ", dim_out)
    print("new_dim_out: ", new_dim_out)
    print("num_heads: ", num_heads)
    print("new_num_heads: ", new_num_heads)
    w = torch.vstack([w_q[0:new_dim_out, :], w_k[0:new_dim_out, :], w_v[0:new_dim_out, :]])
    b = torch.hstack([b_q[0:new_dim_out], b_k[0:new_dim_out], b_v[0:new_dim_out]])
    out_w = out_w[:, 0:new_dim_out]
    return w, b, out_w, out_b

for r in [0.1,0.3,0.5,0.7,0.9]:
    checkpoint_small = checkpoint.copy()
    checkpoint_large = checkpoint.copy()

    it = iter(attn_keys)
    it_out = iter(attn_keys_out)

    for k_w, k_out_w in zip(it, it_out):
        k_b = next(it)
        k_out_b = next(it_out)
        checkpoint_small[k_w], checkpoint_small[k_b], checkpoint_small[k_out_w], checkpoint_small[k_out_b] = mod_small(checkpoint[k_w], checkpoint[k_b], checkpoint[k_out_w], checkpoint[k_out_b], r=r, embed_dim=64)

    print("___________________________________________________")
    it = iter(attn_keys)
    it_out = iter(attn_keys_out)

    for k_w, k_out_w in zip(it, it_out):
        k_b = next(it)
        k_out_b = next(it_out)
        checkpoint_large[k_w], checkpoint_large[k_b], checkpoint_large[k_out_w], checkpoint_large[k_out_b] = mod(checkpoint[k_w], checkpoint[k_b], checkpoint[k_out_w], checkpoint[k_out_b], r=r, embed_dim=64)

    torch.save(checkpoint_small, f"/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b5_static_pruning_mha_{r*100}%_round_off.pth")
    #torch.save(checkpoint_large, f"/beegfs/work/bartels/mmsegmentation/pretrained_weight/segformer/mit_b0_static_pruning_mha_{r*100}%_round_up.pth")