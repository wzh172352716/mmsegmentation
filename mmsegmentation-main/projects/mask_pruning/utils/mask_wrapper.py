import functools
import torch
from torch import nn

from .masks import LearnableMaskConv2d, LearnableMaskLinear, LearnableMask, LearnableMaskMHALinear, LearnableKernelMask
from .mha_mask import LearnableMaskMHAProjection

def rename_parameter(obj, old_name, new_name):
    """
    function for renaming an instance parameter to another name.

    e.g.:
        linear = torch.nn.Linear(1,2)
        linear.weight.shape
        ---> torch.Size([2, 1])
        linear.weight_lin.shape
        ---> AttributeError

        rename_parameter(linear, "weight", "weight_lin")

        linear.weight_lin.shape
        ---> torch.Size([2, 1])
        linear.weight.shape
        ---> AttributeError

    Also works recursively for parameters of parameters, like:
        rename_parameter(linear, "weight.shape", "weight.shape_")

    Args:
        obj: the object where the attribute should be renamed
        old_name: the current name of the parameter
        new_name:the new name of the parameter

    Returns:
        None
    """

    def rename_param(obj, old_name, new_name):
        # print(obj.__dict__.get('_parameters').keys())
        obj.__dict__.get('_parameters')[new_name] = obj._parameters.pop(old_name)

    pre, _, post = old_name.rpartition('.')
    pren, _, postn = new_name.rpartition('.')
    return rename_param(rgetattr(obj, pre) if pre else obj, post, postn)


def rsetattr(obj, attr, val):
    """
    recursive setattr
    Args:
        obj: object
        attr: attribute name as string
        val: value to set the attribute to

    Returns:
        None
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    recursive getattr
    Args:
        obj: object
        attr: attribute name as string
        *args:

    Returns:
        the attribute
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rdelattr(obj, attr, *args):
    """
    recursive delattr
    Args:
        obj: object
        attr: attribute name as string
        *args:

    Returns:

    """

    def _delattr(obj, attr):
        return delattr(obj, attr, *args)

    return functools.reduce(_delattr, [obj] + attr.split('.'))


def mask_class_wrapper(super_class, mode="linear", embedded_dims=64, k=7):
    """
    Function that creates a wrapper class that is a child class of the given super_class.
    Functionality:
        Every time the forward function of the created class is called, every parameter is renamed
        from <param_name> to <param_name> + "_". Then the masks are generated. Every parameter
        is multiplied by their corresponding mask. The result is stored in a parameter, which has
        the original name <param_name>. After that the forward function of the super_class is called.
        As we exchanged all parameters to the masked ones, this function also calculates with the
        masked ones.

        After the forward call of the super_class, all parameters are named back to their original name.

    Args:
        super_class: the super_class
        mode: the mask mode to use ("linear", "conv" or "mha_linear")
        embedded_dims: embedded_dims (only needed for mode="mha_linear")
        k: the exponent for the sigmoid function for mask creation f(x) = 1 / (1 + e^(-k * (x - b))), default = 7

    Returns:
        the created wrapper class that has prunable masks
    """

    def init_masks(self):
        """
        initialize the masks
        Args:
            self:

        Returns:
            None
        """

        self.masks = {}
        # iterate over all parameters of module
        for i, name in enumerate(self.names):
            if "bn." in name:
                continue
            # If current name is weight and the next name in the list is not called bias
            # -> weight with no bias
            if "weight" in name.split(".")[-1] and not (
                    i < len(self.names) - 1 and self.names[i + 1].split(".")[-1] == "bias"):

                # create mask for Conv2d or ConvModule
                if mode == "conv":
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskConv2d(feature_input, feature_output, kernel_size, k=k)

                    # for ConvModule
                    if i < len(self.names) - 2 and "bn.bias" in self.names[i + 2]:
                        self.masks[self.names[i + 2]] = self.masks[name]
                elif mode == "conv_kernel":
                    size_weight = rgetattr(self, name).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableKernelMask(feature_input, feature_output, kernel_size, k=k)

                    # for ConvModule
                    if i < len(self.names) - 2 and "bn.bias" in self.names[i + 2]:
                        self.masks[self.names[i + 2]] = self.masks[name]
                # create mask for MHA layer
                # the order of the parameters in MHA without bias  is (in_proj_weight, out_proj.weight)
                elif mode == "mha_linear" and "out_proj.weight" in name:
                    size_weight = rgetattr(self, name).size()
                    size_bias = size_weight[0]
                    self.masks[name] = LearnableMaskMHALinear(size_weight, int(size_bias),
                                                                  int(size_bias // embedded_dims), k=k)
                    self.masks["in_proj_weight"] = self.masks[name]

            # If current name is bias and the previous name in the list was weight
            # -> weight and bias belong together
            elif "bias" in name.split(".")[-1] and i != 0 and "weight" in self.names[i - 1].split(".")[
                -1]:
                if mode == "linear":
                    size_bias = rgetattr(self, name).size()
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    self.masks[name] = LearnableMaskLinear(size_weight, size_bias, k=k)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "conv":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableMaskConv2d(feature_input, feature_output, kernel_size, k=k)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "conv_kernel":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    feature_output, feature_input, kernel_size, _ = size_weight
                    self.masks[name] = LearnableKernelMask(feature_input, feature_output, kernel_size, k=k)
                    self.masks[self.names[i - 1]] = self.masks[name]
                elif mode == "mha_linear":
                    size_weight = rgetattr(self, self.names[i - 1]).size()
                    size_bias = size_weight[0]

                    if "out_proj.weight" in self.names[i - 1]:
                        self.masks[name] = LearnableMaskMHALinear(size_weight, int(size_bias),
                                                                      int(size_bias // embedded_dims), k=k)
                        self.masks["in_proj_weight"] = self.masks[name]
                        self.masks[self.names[i - 1]] = self.masks[name]
                        self.masks["in_proj_bias"] = self.masks[name]
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented yet")

        self.module_list = nn.ModuleList(self.masks.values())

    def delete_masks(self):
        """
        delete masks
        Args:
            self:

        Returns:

        """
        for mask in self.masks.values():
            del mask
        del self.module_list
        self.masks = {}

    def reinit_masks(self):
        """
        reinit all maks by calling the reinit function from all masks
        Args:
            self:

        Returns:

        """
        with torch.no_grad():
            for i, name in enumerate(self.names):
                if name in self.masks and "weight" in name:
                    mask = self.masks[name]
                    if mode == "conv" or mode == "mha_conv":
                        size_weight = rgetattr(self, name).size()
                        feature_output, feature_input, kernel_size, _ = size_weight
                        mask.reinit(feature_input, feature_output, kernel_size)
                    elif mode == "linear" or (mode == "mha_linear" and "out_proj.weight" in name):
                        size_weight = rgetattr(self, name).size()
                        size_bias = size_weight[0]
                        mask.reinit(size_weight, size_bias)

    def init_wrapper(self, *args, **kw):
        """
        constructor function (__init__)
        calls super constructor and then initializes teh masks
        Args:
            self:
            *args:
            **kw:

        Returns:

        """
        super_class.__init__(self, *args, **kw)
        self.mask_class_wrapper = True
        self.names = [name for name, _ in self.named_parameters()]
        self.init_masks()
        self.training = True

    def reset_parameters_wrapper(self, *args, **kw):
        """
        reset parameters
        Args:
            self:
            *args:
            **kw:

        Returns:

        """
        return super_class._reset_parameters(self, *args, **kw)

    def get_mask_loss(self):
        """
        collects and sums up all partial loss terms from the masks
        Args:
            self:

        Returns:

        """
        loss = 0
        for mask in self.masks.values():
            loss = loss + mask.get_loss()
        return loss

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        wrapper for state_dict function of super_class

        sets the model in state "train" before outputting the state_dict
        and then restores the old model state.
        Args:
            self:
            *args:
            destination:
            prefix:
            keep_vars:

        Returns:

        """
        mode_before = self.training
        self.train()
        res = super_class.state_dict(self, *args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.train(mode_before)
        return res

    def train_wrapper(self, mode_train: bool = True, *args, **kw):
        """
        train wrapper

        During inference (mode_train=False ) it is unnecessary to compute the mask every time new.
        This would just increase the inference time. So this function detects, when the model is changed from
        a) (mode=train -> mode=eval) and b) (mode=eval -> mode=train).

        a) For case a: The masks for every parameter are computed and every parameter is multipled by the mask
        b) For case a: The masks for every parameter are computed and every parameter is divided by the mask



        Args:
            self:
            mode_train:
            *args:
            **kw:

        Returns:

        """
        has_changed = self.training != mode_train
        super_class.train(self, mode=mode_train, *args, **kw)
        if has_changed:
            with torch.no_grad():
                for i, name in enumerate(self.names):
                    if name in self.masks:
                        weight_idx, bias_idx = 0, 1
                        if mode == "mha_linear":
                            weight_idx, bias_idx = (0, 1) if "in_proj" in name else (2, 3)

                        mask = self.masks[name].get_mask()[weight_idx] if "weight" in name else \
                        self.masks[name].get_mask()[bias_idx]
                        if mode_train:
                            mask = torch.where(mask < 0.0001, 0.0, 1.0 / mask)
                        else:
                            mask = torch.where(mask < 0.0001, 0.0, mask)
                        param = rgetattr(self, name)
                        if param.size() != mask.size():
                            print("err")
                            mask = self.masks.pop(name)
                            self.names.pop(i)
                            del mask
                        else:
                            param.copy_(param * mask)
        return self

    def initialize_permutation_if_empty(self, name, index):
        """
        initialize permutations for every mask if not already done
        Args:
            self:
            name:
            index:

        Returns:

        """
        if not self.masks[name].check_permutation_initialized():
            # If current name is weight and next is bias
            # -> weight and bias belong together
            if "in_proj_weight" in name:
                self.masks[name].update_permutation(rgetattr(self, "in_proj_weight_"), rgetattr(self, "in_proj_bias"),
                                                    rgetattr(self, "out_proj.weight"), rgetattr(self, "out_proj.bias"))
            elif "weight" in name.split(".")[-1] and index < len(self.names) - 1 and "bias" in \
                    self.names[index + 1].split(".")[
                        -1]:
                self.masks[name].update_permutation(rgetattr(self, name + "_"),
                                                    rgetattr(self, self.names[index + 1]))
            # If current name is weight and next is not bias
            # -> weight without bias
            elif "weight" in name.split(".")[-1] and not (
                    index < len(self.names) - 1 and "bias" in self.names[index + 1].split(".")[
                -1]):
                self.masks[name].update_permutation(rgetattr(self, name + "_"),
                                                    torch.zeros((rgetattr(self, name + "_").size()[0]),
                                                                device=self.masks[name].get_device()))

    def forward_wrapper(self, *args, **kw):
        """
        forward wrapper

        Every time the forward function of the created class is called, every parameter is renamed
        from <param_name> to <param_name> + "_". Then the masks are generated. Every parameter
        is multiplied by their corresponding mask. The result is stored in a parameter, which has
        the original name <param_name>. After that the forward function of the super_class is called.
        As we exchanged all parameters to the masked ones, this function also calculates with the
        masked ones.

        After the forward call of the super_class, all parameters are named back to their original name.

        Args:
            self:
            *args:
            **kw:

        Returns:

        """
        if self.training:
            last_mask_obj = None
            last_mask = None
            for i, name in enumerate(self.names):
                if name in self.masks:
                    # rename parameter
                    rename_parameter(self, name, name + "_")

                    # init permutation if not initialized already
                    self.initialize_permutation_if_empty(name, i)

                    # get mask
                    weight_idx, bias_idx = 0, 1
                    if mode == "mha_linear":
                        weight_idx, bias_idx = (0, 1) if "in_proj" in name else (2, 3)

                    if self.masks[name] == last_mask_obj:
                        current_mask = last_mask
                    else:
                        current_mask = self.masks[name].get_mask()
                    mask = current_mask[weight_idx] if "weight" in name else current_mask[bias_idx]
                    last_mask_obj = self.masks[name]
                    last_mask = current_mask

                    # create mask parameter with original name
                    rsetattr(self, name, rgetattr(self, name + "_") * mask)

            output = super_class.forward(self, *args, **kw)

            # rename parameters back
            for name in self.names:
                if name in self.masks:
                    rename_parameter(self, name + "_", name)
        else:

            for i, name in enumerate(self.names):
                if name in self.masks:
                    rename_parameter(self, name, name + "_")
                    self.initialize_permutation_if_empty(name, i)
                    rename_parameter(self, name + "_", name)
            output = super_class.forward(self, *args, **kw)

        return output

    return type(f"Wrapper{super_class}", (super_class,), {
        # constructor
        "__init__": init_wrapper,

        # member functions
        "_reset_parameters": reset_parameters_wrapper,
        "get_mask_loss": get_mask_loss,
        "train": train_wrapper,
        "forward": forward_wrapper,
        "initialize_permutation_if_empty": initialize_permutation_if_empty,
        "init_masks": init_masks,
        "delete_masks": delete_masks,
        "reinit_masks": reinit_masks,
        "state_dict": state_dict
    })


def get_p1_values(module):
    """
    get all p1 parameters of all masks
    Args:
        module:

    Returns:

    """
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = p.p1
    return p1_list


def get_num_pruned(module):
    """
    get number of elements that could be pruned
    Args:
        module:

    Returns:

    """
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = (p.p1 * p.lr_mult_factor).int().float()
    return p1_list


def get_percentage_pruned(module):
    """
    get percentage of elements that could be pruned
    Args:
        module:

    Returns:

    """
    p1_list = {}
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            p1_list[n] = (p.p1 * p.lr_mult_factor).int().float() / p.pruning_size
    return p1_list


def get_p1_loss(module):
    """
    get the sum of all mask losses from the model
    Args:
        module:

    Returns:

    """
    loss_sum = 0
    for n, p in module.named_modules():
        if isinstance(p, LearnableMask):
            loss_sum = loss_sum + p.get_loss()
    return loss_sum
