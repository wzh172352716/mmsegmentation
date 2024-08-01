from abc import abstractmethod

from torch import nn
import torch

from .mask_functions import get_permutation_vector, get_weighting_matrix, logistic_function, get_index_matrix


class LearnableMask(nn.Module):
    """
    Base class for the learnable masks for structured pruning
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    @abstractmethod
    def get_device(self):
        """
        Getter for device of mask
        Returns: device

        """
        return "cpu"


class LearnableMaskLinear(LearnableMask):
    """
    Learnable Mask for a linear layer
    """

    def __init__(self, size_weight, size_bias, mask_dim=0, lr_mult_factor=100.0, k=7):
        """

        Args:
            size_weight: size of the weight to mask out as a 2d tensor/tuple of the form (out_features, in_features)
            size_bias: size of the bias as a 1d tensor
            mask_dim: the dimension to mask out, default=0 -> out_features
            lr_mult_factor: the parameter p1 is divided by lr_mult_factor and for generating the mask it is multiplied back by lr_mult_factor,
                            theoretically this parameter should not change anything, but it does.
                            The reason for this is that the effect of weight decay is responsible for this
                            TODO: remove lr_mult_factor by turning off weight decay for p1
            k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7
        """
        super().__init__()
        self.k = k
        # a single parameter which is the only learnable scalar in this class, which can be
        # in the range [-1, pruning_size]
        self.p1 = torch.nn.parameter.Parameter(torch.tensor([-1 / lr_mult_factor], requires_grad=True))
        self.lr_mult_factor = lr_mult_factor
        # self.permutation = None
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.size_together = list(size_weight)

        self.size_together[-1] = self.size_weight[-1] + 1  # because of bias

        self.dim = mask_dim
        self.permutation_initialized = False

        self.pruning_size = self.size_weight[self.dim]
        if self.dim == 1:
            self.non_pruning_size = self.size_together[0]
        else:
            self.non_pruning_size = self.size_together[1]

        # register permutation as this is not learnable, but must be saved with the model
        #print(self.pruning_size)
        self.register_buffer("permutation", torch.zeros(self.pruning_size, device=self.get_device()).int())

    def get_device(self):
        """
        Getter for device of mask
        Returns: device

        """
        return self.p1.device

    def reinit(self, size_weight, size_bias, mask_dim=0):
        """
        Reinit masks after the linear layer was pruned, aka. some neurons have been removed.
        The mask parameter must be reduced by the number of neurons that have been removed from this layer

        Args:
            size_weight: new size of weight in linear layer after pruning - (new_out_features, in_features)
            size_bias: tensor of size new_out_features
            mask_dim: the dimension to mask out, default=0 -> out_features

        Returns:
            None
        """

        # get the number of removed neurons
        dim_diff = self.pruning_size - size_weight[mask_dim]

        # reduce the mask parameter by the amount dim_diff/self.lr_mult_factor
        self.p1.copy_(self.p1 - dim_diff / self.lr_mult_factor)

        # set class parameters to new sizes
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.size_together = list(size_weight)
        self.size_together[-1] = self.size_weight[-1] + 1  # +1 is because of the bias

        self.dim = mask_dim

        self.pruning_size = self.size_weight[self.dim]
        if self.dim == 1:
            self.non_pruning_size = self.size_together[0]
        else:
            self.non_pruning_size = self.size_together[1]

        # fix permutation:
        # if the previous permutation was for example [2,5,1,3,0,4] and the first 3 channels have been removed
        # the resulting permutation should not anymore contain the previous entries for 0,1,2 -> [5,3,4]
        # Then the permutation must be corrected to start again with 0
        if dim_diff > 0 and self.permutation_initialized:
            new_perm = self.permutation - dim_diff
            keep_rows = torch.where(new_perm < 0, 0, 1).nonzero(as_tuple=True)[0].tolist()
            new_perm = new_perm[keep_rows]
            zero_to_n = torch.argsort(torch.zeros_like(new_perm)).to(self.get_device()).to(new_perm.dtype)
            new_perm[torch.argsort(new_perm)] = zero_to_n
            self.permutation = new_perm


    def check_permutation_initialized(self):
        """
        Check whether the permutation is already initialized.
        The permutation should not be initialized twice, because this would change the output

        Returns:
            True if initialized
        """
        if not self.permutation_initialized and self.permutation.sum() == 0:
            return False
        elif not self.permutation_initialized:
            self.permutation = self.permutation.int()
            self.permutation_initialized = True
            return True
        elif self.permutation_initialized:
            return True

    def update_permutation(self, weight, bias):
        """
        updates the permutation

        Args:
            weight: weight of the linear layer with shape (out_features, in_features)
            bias: bias of the linear layer with shape (out_features)

        Returns:
            return permutation vector with shape (out_features)
        """
        if not self.permutation_initialized:
            self.permutation_initialized = True

        # create one single vector of shape (out_features, in_features + 1)
        wb = torch.hstack([weight, torch.unsqueeze(bias, -1)])

        # get permutation
        self.permutation = get_permutation_vector(wb)

    def get_loss(self):
        """
        Simple loss function: loss = -p1

        To avoid p1 from getting unnecessary large we introduced a min term to prevent p1 from getting 5 larger
        than the number of neurons in this layer:
            loss = -min(p1, (self.pruning_size+5) / self.lr_mult_factor)
        Returns:
            loss
        """
        return - torch.min(self.p1, torch.tensor([(self.pruning_size + 5) / self.lr_mult_factor], device=self.get_device()))

    def get_mask(self):
        """
        get the mask based in the current parameter p1
        Returns:
            weighting mask
        """
        #print(self.p1 * self.lr_mult_factor)
        #print("In Mask: ", self.permutation)
        mask = get_weighting_matrix(torch.min(self.p1 * self.lr_mult_factor, torch.tensor([self.pruning_size + 5], device=self.get_device())),
                                    self.pruning_size, self.non_pruning_size, k=self.k, device=self.get_device())[self.permutation]
        mask_weight = mask[:, 0:-1]
        mask_bias = mask[:, -1]
        return mask_weight, mask_bias


class LearnableMaskConv2d(LearnableMaskLinear):
    """
    Learnable Mask for a Conv2d layer

    Basically very similar to the LearnableMaskLinear class, because the Conv2d
    layer is reshaped to work with the LearnableMaskLinear.

    The corresponding Conv2d(in_channels, out_channels, (kernel_size_x, kernel_size_y)) layer
    has a weight of shape (out_channels, in_channels, kernel_size_x, kernel_size_y). We want
    to mask out complete output channels, so we can reshape this weight
    to (out_channels, in_channels*kernel_size_x*kernel_size_y). Then we can easily apply again the same
    mask as for the LearnableMaskLinear for size size_weight=(out_channels, in_channels*kernel_size_x*kernel_size_y)
    """
    def __init__(self, input_features, output_features, kernel_size, lr_mult_factor=100.0, k=7):
        """

        Args:
            input_features: number input_features of Conv layer
            output_features: number output_features of Conv layer
            kernel_size: the kernel size of Conv layer
            lr_mult_factor: lr_mult_factor: the parameter p1 is divided by lr_mult_factor and for generating the mask it is multiplied back by lr_mult_factor,
                            theoretically this parameter should not change anything, but it does.
                            The reason for this is that the effect of weight decay is responsible for this
                            TODO: remove lr_mult_factor by turning off weight decay for p1
            k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7
        """
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().__init__(size_weight, size_bias, mask_dim=0, lr_mult_factor=lr_mult_factor, k=k)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def reinit(self, input_features, output_features, kernel_size):
        """
        Reinit masks after the linear layer was pruned, aka. some feature maps have been removed.
        The mask parameter must be reduced by the number of feature maps that have been removed from this layer
        Args:
            input_features: number of input_features
            output_features: new number of output_features
            kernel_size: kernel size

        Returns:
            None
        """

        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        super().reinit(size_weight, size_bias)
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size

    def update_permutation(self, weight, bias):
        """
        updates the permutation by calling the update_permutation function from the base class
        Args:
            weight:
            bias:

        Returns:

        """
        weight_shallow = torch.reshape(weight, (
            self.output_features, self.input_features * self.kernel_size * self.kernel_size))
        super().update_permutation(weight_shallow, bias)

    def get_mask(self):
        """
        get the mask based in the current parameter p1 by calling the get_mask function from the base class
        and reshaping it into the needed shape for the Conv2d layer
        Returns:
            weighting mask
        """
        mask_weight, mask_bias = super().get_mask()
        return torch.reshape(mask_weight,
                             (self.output_features, self.input_features, self.kernel_size, self.kernel_size)), mask_bias



class LearnableMaskMHA(LearnableMaskLinear):
    def __init__(self, input_features, output_features, kernel_size, num_heads, lr_mult_factor=100.0, k=7):
        size_weight = (output_features // num_heads, kernel_size * kernel_size * input_features)
        size_bias = output_features // num_heads
        super().__init__(size_weight, size_bias // num_heads, mask_dim=0, lr_mult_factor=lr_mult_factor, k=k)
        self.input_features = input_features
        self.output_features = output_features // num_heads
        self.kernel_size = kernel_size
        self.num_heads = num_heads

    def reinit(self, input_features, output_features, kernel_size):
        size_weight = (output_features // self.num_heads, kernel_size * kernel_size * input_features)
        size_bias = output_features // self.num_heads
        super().reinit(size_weight, size_bias // self.num_heads)
        self.input_features = input_features
        self.output_features = output_features // self.num_heads
        self.kernel_size = kernel_size

    def update_permutation(self, weight, bias):
        weight_shallow = torch.abs(weight.reshape(self.num_heads, self.output_features,
                                                  self.input_features * self.kernel_size * self.kernel_size)).sum(dim=0)
        bias_shallow = torch.abs(bias.reshape(self.num_heads, self.output_features)).sum(dim=0)
        super().update_permutation(weight_shallow, bias_shallow)

    def get_mask(self):
        mask_weight, mask_bias = super().get_mask()
        r = torch.reshape(mask_weight,
                          (self.output_features, self.input_features, self.kernel_size, self.kernel_size))
        weight_mask_r = r.expand(self.num_heads, self.output_features, self.input_features, self.kernel_size,
                                 self.kernel_size).reshape(self.num_heads * self.output_features, self.input_features,
                                                           self.kernel_size, self.kernel_size)
        bias_mask_r = mask_bias.expand(self.num_heads, self.output_features).reshape(
            self.num_heads * self.output_features)
        return weight_mask_r, bias_mask_r


class LearnableMaskMHALinear(LearnableMaskLinear):
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

    It is only possible to mask out a complete embed_dim dimension in every head of the num_heads.
    So there are embed_dim//num_heads elements that can be masked out.

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
        # We handle the mask creation of the MHA layer as a LinearLearnableMask of size (embed_dim//num_heads, embed_dim)
        size_weight = (output_features // num_heads, input_features)
        self.num_heads = num_heads
        super().__init__(size_weight, size_bias // num_heads, mask_dim, lr_mult_factor, k=k)

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
        size_weight = (output_features // self.num_heads, input_features)
        size_bias = int(size_bias) // self.num_heads
        super().reinit(size_weight, size_bias)


    def update_permutation(self, weight_in, bias_in, weight_out, bias_out):
        """
        updates the permutation by calling the update_permutation function from the base class
        Args:
            weight_in: the in_proj_weight of the MHA layer. Shape (3*embed_dim, embed_dim)
            bias_in: the in_proj_bias of the MHA layer. Shape (3*embed_dim)
            weight_out: the out_proj_weight of the MHA layer. Shape (embed_dim, embed_dim)
            bias_out: the out_proj_bias of the MHA layer. Shape (embed_dim)

        Returns:

        """
        embed_dim = self.pruning_size * self.num_heads
        # weight_in (3*embed_dim, embed_dim) is first reshaped to (3, embed_dim, embed_dim) and then summed up over
        # the first dim -> (embed_dim, embed_dim)
        # then the result is reshaped to (num_heads, embed_dim//num_heads, embed_dim) and then summed up again over
        # the first dim -> (embed_dim//num_heads, embed_dim)
        weight_shallow_in = torch.abs(torch.abs(weight_in.reshape(3, embed_dim, embed_dim)).sum(dim=0).reshape(
            self.num_heads, self.pruning_size, embed_dim)).sum(dim=0)

        # weight_out (embed_dim, embed_dim)) is reshaped to (num_heads, embed_dim//num_heads, embed_dim) and then summed up
        # over the first dim -> (embed_dim//num_heads, embed_dim)
        weight_shallow_out = torch.abs(
            weight_out.reshape(self.num_heads, self.pruning_size, embed_dim)).sum(dim=0)

        # bias_in (3*embed_dim) is first reshaped to (3, embed_dim) and then summed up over
        # the first dim -> (embed_dim)
        # then the result is reshaped to (num_heads, embed_dim//num_heads) and then summed up again over
        # the first dim -> (embed_dim//num_heads)
        bias_shallow_in = torch.abs(torch.abs(bias_in.reshape(3, self.num_heads*self.pruning_size)).sum(dim=0).reshape(self.num_heads, self.pruning_size)).sum(dim=0)

        # bias_out (embed_dim)) is reshaped to (num_heads, embed_dim//num_heads) and then summed up
        # over the first dim -> (embed_dim//num_heads)
        bias_shallow_out = torch.abs(bias_out.reshape(self.num_heads, self.pruning_size)).sum(dim=0)

        # call update_permutation from parent class with calculated sums
        super().update_permutation(weight_shallow_in+weight_shallow_out, bias_shallow_in+bias_shallow_out)

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

        embed_dim = self.non_pruning_size-1

        # expand mask_weight from (embed_dim//num_heads, embed_dim) to (embed_dim, embed_dim) by repeating it num_heads times
        out_proj_weight_mask = mask_weight.expand(self.num_heads, self.pruning_size, embed_dim).reshape(embed_dim, embed_dim)
        # for the out_proj_weight both dimensions need to be multiplied by the mask, so we need the transpose
        transpose = out_proj_weight_mask.permute(1, 0)
        out_proj_weight_mask = out_proj_weight_mask * transpose

        # expand mask_bias from (embed_dim//num_heads) to (embed_dim) by repeating it num_heads times
        out_proj_bias_mask = mask_bias.expand(self.num_heads, self.pruning_size).reshape(embed_dim)


        # expand out_proj_weight_mask from (embed_dim, embed_dim) to (3, embed_dim, embed_dim) by repeating it 3 times
        in_proj_weight_mask = out_proj_weight_mask.expand(3, embed_dim, embed_dim).reshape(3 * embed_dim, embed_dim)

        # expand out_proj_bias_mask from (embed_dim) to (3, embed_dim) by repeating it 3 times
        in_proj_bias_mask = out_proj_bias_mask.expand(3, embed_dim).reshape(3 * embed_dim)
        return in_proj_weight_mask, in_proj_bias_mask, out_proj_weight_mask, out_proj_bias_mask


class LearnableKernelMask(LearnableMask):
    def get_weighting_matrix(self, b, feature_maps, pruning_dim, non_pruning_dim, k=7):
        b_expand = b.unsqueeze(1).unsqueeze(1).expand((feature_maps, pruning_dim, non_pruning_dim))
        index_mat = self.get_index_matrix(pruning_dim, non_pruning_dim)
        return logistic_function(index_mat.unsqueeze(0).expand(feature_maps, pruning_dim, non_pruning_dim), k, b_expand)

    def get_index_matrix(self, rows, colums):
        if rows % 2 == 1:
            arr = torch.zeros((rows, colums), requires_grad=False, device=self.get_device()).to(torch.float)
            for i in range(rows):
                arr[i] = (rows // 2 - abs(rows // 2 - i)) * 2
            for i in range(rows - rows // 2 - 1):
                arr[i] = arr[i] + 1
            return arr
        else:
            arr = self.get_index_matrix(rows + 1, colums)[:-1, :]
            for i in range(rows - rows // 2):
                arr[i + rows // 2] = arr[i + rows // 2] - 2
            return arr

    def __init__(self, input_features, output_features, kernel_size, lr_mult_factor=100.0, loss_factor=10.0, k=7):
        super().__init__()
        size_weight = (output_features, kernel_size * kernel_size * input_features)
        size_bias = output_features
        self.k = k
        self.p1 = torch.nn.parameter.Parameter(-1 * torch.ones((output_features * 2)) / lr_mult_factor,
                                               requires_grad=True)
        self.lr_mult_factor = lr_mult_factor
        self.size_weight = size_weight
        self.size_bias = size_bias
        self.input_features = input_features
        self.output_features = output_features
        self.kernel_size = kernel_size
        self.loss_factor = loss_factor
        self.permutation = True
        self.permutation_initialized = True

    def check_permutation_initialized(self):
        return True

    def update_permutation(self, weight, bias):
        pass

    def get_loss(self):
        #return - torch.sum(self.p1)  # / (self.kernel_size**2) * self.loss_factor
        return - torch.sum(torch.min(self.p1, torch.ones_like(self.p1, device=self.get_device()) * ((self.kernel_size + 5) / self.lr_mult_factor)))

    def get_mask(self):
        b_x = self.p1[0:self.p1.size(0) // 2] * self.lr_mult_factor
        b_y = self.p1[self.p1.size(0) // 2:] * self.lr_mult_factor
        m_x = self.get_weighting_matrix(b_x, self.output_features, self.kernel_size,
                                        self.kernel_size * self.input_features, k=self.k).reshape(
            (self.output_features, self.kernel_size, self.input_features, self.kernel_size)).permute((0, 2, 1, 3))
        m_y = self.get_weighting_matrix(b_y, self.output_features, self.kernel_size,
                                        self.kernel_size * self.input_features, k=self.k).reshape(
            (self.output_features, self.kernel_size, self.input_features, self.kernel_size)).permute((0, 2, 3, 1))

        mask = m_x * m_y
        mask_bias = mask[:, 0, (self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2]
        return mask, mask_bias

    def get_pruned_kernel_sizes(self):
        sizes = {}
        for kernel_idx in range(self.output_features):
            x = self.p1[0:self.p1.size(0) // 2][kernel_idx]
            y = self.p1[self.p1.size(0) // 2:][kernel_idx]
            x_size_pruned = self.kernel_size - max(x.int(), 0)
            y_size_pruned = self.kernel_size - max(y.int(), 0)
            sizes[(int(x_size_pruned), int(y_size_pruned))] = sizes.get((int(x_size_pruned), int(y_size_pruned)), 0) + 1
        return sizes


    @abstractmethod
    def get_device(self):
        """
        Getter for device of mask
        Returns: device

        """
        return self.p1.device