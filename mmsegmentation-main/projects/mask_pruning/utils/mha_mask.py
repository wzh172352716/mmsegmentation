import torch

from .mask_wrapper import LearnableMaskLinear


class LearnableMaskMHAProjection(LearnableMaskLinear):
    """
    Learnable Mask for a Multihead attention layer

    Basically very similar to the LearnableMaskLinear class, because the MultiHeadAttention
    layer is reshaped to work with the LearnableMaskLinear.

    The corresponding MultiHeadAttention(embed_dim, num_heads) layer
    has weights of shape:
        in_proj_weight:     (3*embed_dim, embed_dim)
        in_proj_bias:       (3*embed_dim)
        out_proj_weight:    (embed_dim, embed_dim)
        out_proj_bias:      (embed_dim)
    """

    def __init__(self, size_weight, size_bias, num_heads, mask_dim=0, lr_mult_factor=100.0, k=7):
        """

        Args:
            size_weight: the size of the weight of the corresponding MultiHeadAttention layer in (embed_dim, embed_dim)
            size_bias: the size of the bias of the corresponding MultiHeadAttention layer (embed_dim)
            num_heads: the number of heads in the corresponding MultiHeadAttention layer
            mask_dim: must be 0 and is just to match signature of base class
            lr_mult_factor: lr_mult_factor: the parameter p1 is divided by lr_mult_factor and for generating the mask it is multiplied back by lr_mult_factor,
                            theoretically this parameter should not change anything, but it does.
                            The reason for this is that the effect of weight decay is responsible for this
                            TODO: remove lr_mult_factor by turning off weight decay for p1
            k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7
        """
        output_features, input_features = size_weight
        # We handle the mask creation of the MHA layer as two LinearLearnableMasks
        # For input projection one Mask of size (embed_dim, embed_dim)
        size_weight = (output_features, input_features)
        super().__init__(size_weight, size_bias, mask_dim, lr_mult_factor, k=k)

    def reinit(self, size_weight, size_bias, mask_dim=0):
        """
        Reinit masks after the MHA layer was pruned, aka. some embedded dimensions have been removed.
        The mask parameter must be reduced by the number of embedded dimensions that have been removed from this layer
        Args:
            size_weight: (embed, embed)
            size_bias: (embed)
            mask_dim: must be 0 and is just to match signature of base class

        Returns:
            None
        """
        output_features, input_features = size_weight
        size_weight = (output_features, input_features)
        size_bias = int(size_bias)
        super().reinit(size_weight, size_bias)

    def update_permutation(self, weight_in, bias_in, weight_out, bias_out):
        """
        Updates the permutation. For this class the permutation is just a list reaching from 0 to embed_dim-1.
        A different permutation is difficult to employ here because, later during inference it is not possible
        to have heads with different sizes. So only completely pruned heads can be removed and there will be max.
        1 partially pruned head.
        Args:
            weight_in: the in_proj_weight of the MHA layer. Shape (3*embed_dim, embed_dim)
            bias_in: the in_proj_bias of the MHA layer. Shape (3*embed_dim)
            weight_out: the out_proj_weight of the MHA layer. Shape (embed_dim, embed_dim)
            bias_out: the out_proj_bias of the MHA layer. Shape (embed_dim)

        Returns:

        """
        if not self.permutation_initialized:
            self.permutation_initialized = True
        self.permutation = torch.arange(self.pruning_size)

    def get_mask(self):
        """
        get the mask based in the current parameter p1 by calling the get_mask function from the base class
        and reshaping it into the needed shape for the MHA layer
        Returns:
            weighting mask
        """
        mask_weight, mask_bias = super().get_mask()
        # mask_weight has shape (embed_dim//num_heads, embed_dim)
        # mask_bias has shape (embed_dim//num_heads)

        embed_dim = self.non_pruning_size - 1

        # for the out_proj_weight both dimensions need to be multiplied by the mask, so we need the transpose
        out_proj_weight_mask = mask_weight.permute(1, 0)

        # expand mask_bias from (embed_dim//num_heads) to (embed_dim) by repeating it num_heads times
        out_proj_bias_mask = torch.ones_like(mask_bias)

        # expand out_proj_weight_mask from (embed_dim, embed_dim) to (3, embed_dim, embed_dim) by repeating it 3 times
        in_proj_weight_mask = mask_weight.expand(3, embed_dim, embed_dim).reshape(3 * embed_dim, embed_dim)

        # expand mask_bias from (embed_dim) to (3, embed_dim) by repeating it 3 times
        in_proj_bias_mask = mask_bias.expand(3, embed_dim).reshape(3 * embed_dim)
        return in_proj_weight_mask, in_proj_bias_mask, out_proj_weight_mask, out_proj_bias_mask
