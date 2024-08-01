import math
from typing import Optional, Tuple

import numpy as np
import torch
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.print_helper import _format_size
from torch import nn, functional, Tensor
from torch.nn import Parameter
from torch.nn.functional import softmax, linear, _mha_shape_check
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear



class ModifiedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.bias = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = False
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=self.bias, **factory_kwargs)
        # self.dgl_layer_out = DGLLayer(embed_dim, embed_dim, top_r=1.0)
        self._reset_parameters()

    def set_topr_dgl(self, topr):
        assert 0.0 <= topr <= 1.0, f"the topr value must be between 0 and 1, but got {topr}"
        self.dgl_layer_q.set_topr(topr)
        self.dgl_layer_k.set_topr(topr)
        self.dgl_layer_v.set_topr(topr)
        # self.dgl_layer_out.set_topr(topr)

    def _inweight_projection(self, q, k, v, w, b):
        w_qk, _, w_v = w.chunk(3)
        if b is None:
            b_qk = b_v = None
        else:
            b_qk, _, b_v = b.chunk(3)
        return linear(q, w_qk, b_qk), linear(k, w_qk, b_qk), linear(v, w_v, b_v)

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                average_attn_weights=True) -> Tuple[Tensor, Optional[Tensor]]:
        # print(key.size(), query.dim(), self.batch_first)
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        is_batched = _mha_shape_check(query, key, value, None, None, self.num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q, k, v = self._inweight_projection(query, key, value, self.in_proj_weight, self.in_proj_bias)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))


        if not is_batched:
            attn_output = attn_output.squeeze(1)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        else:
            return attn_output, None
