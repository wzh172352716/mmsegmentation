import functools
import os
import pickle
from typing import Any, Optional, Union

import mmcv
import torch
from mmcv.cnn import ConvModule
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from torch.nn.modules.module import register_module_forward_pre_hook

from ..backbones.mit_prunable import EfficientMultiheadAttention_Conv2d_pruned, MixFFN_Conv2d_pruned
from ..utils import set_identity_layer_mode
from ..utils.identity_conv import EmptyModule, DummyModule
from ..utils import LearnableMask, LearnableKernelMask, LearnableMaskLinear, \
    LearnableMaskConv2d, rgetattr, LearnableMaskMHA, rsetattr
import logging
from mmengine.logging import print_log
import torch_pruning as tp
import os.path as osp

import traceback
import sys

from mmseg.structures import SegDataSample
from ..utils.mask_functions import get_weighting_matrix

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class MaskPruningHook(Hook):
    """
    """
    priority = 'LOW'

    def __init__(self, do_logging=True, do_explicit_pruning=True, logging_interval=100, pruning_interval=25,
                 debug=True, prune_at_start=False):
        self.logging_interval = logging_interval
        self.pruning_interval = pruning_interval
        self.logging = do_logging
        self.do_explicit_pruning = do_explicit_pruning
        self.remove_total = 0
        self.num_weights_total_first = -1
        self.model_sizes_org = {}
        self.history = []
        self.debug = debug
        self.prune_at_start = prune_at_start

    def find_last_history(self, path):
        save_file = osp.join(path, 'last_history')
        if os.path.exists(save_file):
            return save_file
        else:
            print_log('Did not find last_history to be resumed.')
            return None

    def get_data_batch(self, input_shape):
        result = {}

        result['ori_shape'] = input_shape[-2:]
        result['pad_shape'] = input_shape[-2:]
        result['img_shape'] = input_shape[-2:]
        data_batch = {
            'inputs': [torch.rand(input_shape)],
            'data_samples': [SegDataSample(metainfo=result)]
        }
        return data_batch

    def is_resume(self, runner):
        return runner._resume or runner._load_from

    def before_run(self, runner) -> None:
        history_filename = self.find_last_history(runner.work_dir)
        if history_filename is not None and self.is_resume(runner):
            # print(f"load {history_filename}")
            with open(history_filename, 'rb') as handle:
                history = pickle.load(handle)

            set_identity_layer_mode(runner.model, True)

            def _forward_func(self, data):
                return self.test_step(data)

            def _output_transform(data):
                l = []
                for i in data:
                    l.append(i.seg_logits.data)
                return l

            data_batch = runner.model.module.data_preprocessor(self.get_data_batch((3, 512, 512)))
            DG = tp.DependencyGraph().build_dependency(runner.model,
                                                       example_inputs=data_batch,
                                                       forward_fn=_forward_func,
                                                       output_transform=_output_transform)
            DG.load_pruning_history(history)
            self.history.extend(history)
            for module_name, module in runner.model.named_modules():
                if getattr(module, "mask_class_wrapper", False):
                    module.reinit_masks()
            set_identity_layer_mode(runner.model, False)

    def init_model_stats(self, model):
        for n, p in model.named_modules():
            if isinstance(p, LearnableKernelMask):
                pass
            elif isinstance(p, LearnableMask):
                structures_per_pruned_instance = p.non_pruning_size
                max_structures = p.pruning_size
                self.model_sizes_org[n] = (structures_per_pruned_instance, max_structures)

    def print_pruning_stats(self, model):
        num_weights_pruned_total = 0
        num_weights_total = 0
        for n, p in model.named_modules():
            if isinstance(p, LearnableKernelMask):
                sizes = p.get_pruned_kernel_sizes()
                print_log(
                    f"{n}: {sizes} kernels",
                    logger='current',
                    level=logging.INFO)
            elif isinstance(p, LearnableMask):
                num_pruned = int((p.p1 * p.lr_mult_factor).int())
                structures_per_pruned_instance = self.model_sizes_org.get(n, [1])[0]  # p.non_pruning_size
                max_structures = self.model_sizes_org.get(n, [1, 1])[1]  # p.pruning_size
                num_pruned = min(num_pruned, max_structures) + self.model_sizes_org.get(n, [1, 1])[1] - p.pruning_size
                num_weights_pruned_total += num_pruned * structures_per_pruned_instance
                num_weights_total += max_structures * structures_per_pruned_instance

                print_log(
                    f"{n}: {num_pruned}/{max_structures} elements pruned; "
                    f"{num_pruned * structures_per_pruned_instance}/{max_structures * structures_per_pruned_instance} weight pruned "
                    f"({100.0 * num_pruned / max_structures}%) {p.pruning_size}x{p.non_pruning_size}",
                    logger='current',
                    level=logging.INFO)

        pytorch_total_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not "identity_conv" in n)
        print_log("\n_____________________________\n"
                  f"Total weights pruned {self.num_weights_total_first - pytorch_total_params}/{self.num_weights_total_first}"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.INFO)

    def _get_pruning_type(self, module):
        if isinstance(module, torch.nn.Linear):
            return tp.prune_linear_out_channels
        elif isinstance(module, torch.nn.Conv2d):
            return tp.prune_conv_out_channels
        elif isinstance(module, ConvModule):
            return tp.prune_conv_out_channels
        else:
            return tp.prune_multihead_attention_out_channels

    def remove_empty_modules(self, model, module, module_name):
        remove_indexes = self._get_remove_indexes(module)

        if self.debug and len(remove_indexes) >= list(module.masks.values())[0].pruning_size:
            print("remove something")
            self._remove_module_completely(module_name, model)

    def _remove_module_completely(self, module_name, model):

        if "ffn" in module_name:
            #if "layers.3.1.1.ffn.layers.0" in module_name:
            module_name = module_name[:module_name.index(".ffn.")+4]
            #print(f"remove completely {module_name} by DummyModule")
            rsetattr(model, module_name, DummyModule())
            self.history.append(f"deleteffn {module_name}")
            return
        print(f"remove completely {module_name} by Empty")
        rsetattr(model, module_name, EmptyModule())
        self.history.append(f"delete {module_name}")

    def _reinit_masks(self, model):
        for name, m in model.named_modules():
            if getattr(m, "mask_class_wrapper", False):
                m.reinit_masks()

    def _get_remove_indexes(self, module):
        mask_bias_soft = list(module.masks.values())[0].get_mask()[-1]
        mask_bias = torch.where(mask_bias_soft < 0.001, 1, 0)
        return mask_bias.nonzero(as_tuple=True)[0].tolist()

    def _transfer_bias_fc1(self, index, pe_conv, fc2):
        relu = torch.nn.GELU()
        mask_bias_soft = list(pe_conv.masks.values())[0].get_mask()[-1]
        #print(mask_bias_soft)
        bias_relu = relu(pe_conv.bias * mask_bias_soft).unsqueeze(-1).unsqueeze(-1)
        z = torch.zeros_like(bias_relu)
        z[index] = bias_relu[index]
        bias_offset = torch.nn.functional.conv2d(z, fc2.weight)
        #print("bias_offset:", bias_offset)
        fc2.bias = torch.nn.Parameter(fc2.bias + bias_offset[:, 0, 0])

    def _get_non_intersecting(self, perm1, perm1_num, perm2, perm2_num):
        idx1 = perm1.tolist() if isinstance(perm1, torch.Tensor) else perm1
        idx2 = perm2.tolist() if isinstance(perm2, torch.Tensor) else perm2

        idx1_part = [idx1.index(i) for i in range(perm1_num)]
        idx2_part = [idx2.index(i) for i in range(perm2_num)]
        non_intersecting = [item for item in idx1_part if item not in idx2_part]
        #print(idx1_part)
        #print(idx2_part)
        #print(non_intersecting)
        #
        #print("_______________")

        for i, num in enumerate(non_intersecting):
            # switch idx: idx2.index(i + perm2_num + 1), with idx: idx2.index(i + perm2_num)
            for j in range(perm2_num + i, -1, -1):
                idxa, idxb = idx2.index(j), idx2.index(j + 1)
                idx2[idxa], idx2[idxb] = j + 1, j

            # switch idx: num, with idx: idx2.index(i + perm2_num)
            idx2[idx2.index(0)] = idx2[num]
            idx2[num] = 0

        return idx2, perm2_num + len(non_intersecting)

    def _prune_mixffn_fc1(self, module_name, module, remove_indexes, model):
        # get mixffn pe_conv
        pe_conv_name = module_name[:module_name.index("ffn.layers.0.conv") + 11] + "2.conv"
        pe_conv = rgetattr(model, pe_conv_name)
        fc2_name = module_name[:module_name.index("ffn.layers.0.conv") + 11] + "5.conv"
        fc2 = rgetattr(model, fc2_name)
        rem_indices_pe = self._get_remove_indexes(pe_conv)

        new_perm_pe, len_rem_indices = self._get_non_intersecting(list(module.masks.values())[0].permutation,
                                                                  len(remove_indexes),
                                                                  list(pe_conv.masks.values())[0].permutation,
                                                                  len(rem_indices_pe))
        # print(new_perm_pe, len_rem_indices)
        mask_class = list(pe_conv.masks.values())[0]
        #
        diff = len_rem_indices - len(rem_indices_pe)
        # print(new_perm_pe[:diff])

        for idx in [new_perm_pe.index(i) for i in range(diff)]:
            # print(idx)
            mod_w = torch.tensor(pe_conv.weight, device=pe_conv.weight.device)
            mod_w[idx] = 0
            pe_conv.weight = torch.nn.Parameter(mod_w)


        for idx in [new_perm_pe.index(i) for i in range(diff)]:
            self._transfer_bias_fc1(idx, pe_conv, fc2)
            mod_b = torch.tensor(pe_conv.bias, device=pe_conv.bias.device)
            mod_b[idx] = 0
            pe_conv.bias = torch.nn.Parameter(mod_b)
            # after = mixffn(inp)
            # print("diff:", torch.sum(torch.abs(before - after)))

        #################################################################################################


        mask_class.permutation = torch.tensor(new_perm_pe, device=mask_class.permutation.device)
        mask_class.p1 = torch.nn.Parameter(mask_class.p1 + diff / mask_class.lr_mult_factor)
        model.train()
        model.eval()

        #####################################################################################################
        new_perm_fc1, len_rem_indices_fc1 = self._get_non_intersecting(list(pe_conv.masks.values())[0].permutation,
                                                                  len(rem_indices_pe), list(module.masks.values())[0].permutation,
                                                                  len(remove_indexes))
        diff_fc1 = len_rem_indices_fc1 - len(remove_indexes)
        mask_class_fc1 = list(module.masks.values())[0]
        mask_class_fc1.permutation = torch.tensor(new_perm_fc1, device=mask_class_fc1.permutation.device)
        mask_class_fc1.p1 = torch.nn.Parameter(mask_class_fc1.p1 + diff_fc1 / mask_class_fc1.lr_mult_factor)
        model.train()
        model.eval()

        #print(remove_indexes)
        #print(rem_indices_pe)
        remove_indexes.extend(rem_indices_pe)
        return list(set(remove_indexes))
        """rem_idxs = list(set(remove_indexes.extend(rem_indices_pe)))
        diff_fc1 = len(rem_idxs) - remove_indexes
        list(module.masks.values())[0].p1 = torch.nn.Parameter(list(module.masks.values())[0].p1 + diff_fc1 / list(module.masks.values())[0].lr_mult_factor)
        return rem_idxs"""
    def _prune_module(self, model, module, module_name, DG, _forward_func, _output_transform):
        remove_indexes = self._get_remove_indexes(module)
        max_size = list(module.masks.values())[0].pruning_size
        if len(remove_indexes) >= max_size:
            remove_indexes = remove_indexes[:-1]

        if len(remove_indexes) > 0:
            if "ffn.layers.2.conv" in module_name:
                print("ffn.layers.2.conv", remove_indexes)
                #print("ffn.layers.2.conv", torch.sum(torch.abs(module.weight[torch.tensor(remove_indexes)])))
                #print("ffn.layers.2.conv", torch.sum(torch.abs(module.bias[torch.tensor(remove_indexes)])))
                #return 0

            if "ffn.layers.0.conv" in module_name:
                remove_indexes = self._prune_mixffn_fc1(module_name, module, remove_indexes, model)

            if isinstance(module, ConvModule):
                module_to_prune = module.conv
            else:
                module_to_prune = module
            tp_type = self._get_pruning_type(module)

            try:
                group = DG.get_pruning_group(module_to_prune, tp_type, idxs=remove_indexes)
            except Exception as e:
                print(e)
                return 0

            if DG.check_pruning_group(group) and len(remove_indexes) > 0:
                """if "ffn.layers.0.conv" in module_name:
                    print(group)
                    print([item.idxs for item in group])"""
                model.eval()
                group.prune()
                self._reinit_masks(model)
                return len(remove_indexes)
            else:
                return 0
        else:
            return 0

    """
    Checks weather the given module should be pruned. Return true if yes otherwise false
    """

    def _check_if_prunable(self, module):
        res_bool = True
        # Is a wrapper class
        res_bool &= getattr(module, "mask_class_wrapper", False)
        if not res_bool:
            return False

        first_mask = list(module.masks.values())[0]

        # has at least one mask
        res_bool &= len(list(module.masks.values())) > 0
        #
        is_conv = isinstance(module, torch.nn.Conv2d) or isinstance(module, ConvModule)
        masks_are_convs = isinstance(first_mask, LearnableMaskConv2d) or isinstance(first_mask, LearnableMaskMHA)

        is_linear = isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.MultiheadAttention)
        masks_are_linear = isinstance(first_mask, LearnableMaskLinear)

        module_type_bool = (is_conv and masks_are_convs) or (is_linear and masks_are_linear)

        return res_bool and module_type_bool

    def prune_weight(self, model):
        if hasattr(model, 'data_preprocessor'):
            data_batch = model.data_preprocessor(self.get_data_batch((3, 512, 512)))
        else:
            data_batch = model.module.data_preprocessor(self.get_data_batch((3, 512, 512)))
        set_identity_layer_mode(model, True)

        def _forward_func(self, data):
            return self.test_step(data)

        def _output_transform(data):
            l = []
            for i in data:
                l.append(i.seg_logits.data)
            return l

        num_removed = 0
        # Remove all completely empty modules
        for module_name, module in model.named_modules():
            if self._check_if_prunable(module):
                self.remove_empty_modules(model, module, module_name)

        DG = tp.DependencyGraph().build_dependency(model, example_inputs=data_batch, forward_fn=_forward_func,
                                                   output_transform=_output_transform)
        for module_name, module in model.named_modules():
            if self._check_if_prunable(module):
                num_removed += self._prune_module(model, module, module_name, DG, _forward_func, _output_transform)

        self.history.extend(DG.pruning_history())
        set_identity_layer_mode(model, False)
        self.remove_total += num_removed
        print_log("\n_____________________________\n"
                  f"Total weights removed {self.remove_total} (+ {num_removed})"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.DEBUG)

    def save_history(self, runner):
        history_filename = osp.join(runner.work_dir, f"iter_{runner.iter + 1}.history")
        history_filename_last = osp.join(runner.work_dir, f"last_history")
        with open(history_filename, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(history_filename_last, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:

        if self.num_weights_total_first == -1:
            pytorch_total_params = sum(p.numel() for n, p in runner.model.named_parameters() if p.requires_grad and not "identity_conv" in n)
            self.num_weights_total_first = pytorch_total_params
            self.init_model_stats(runner.model)
            print_log("\n_____________________________\n"
                      f"Total weights of model {self.num_weights_total_first}"
                      "\n_____________________________\n",
                      logger='current',
                      level=logging.INFO)

        if (self.prune_at_start and runner.iter == 50) or (self.do_explicit_pruning and (
                self.every_n_train_iters(runner, self.pruning_interval) or self.is_last_train_iter(runner))):
            self.prune_weight(runner.model)

        if self.logging and (
                self.every_n_train_iters(runner, self.logging_interval) or self.is_last_train_iter(runner)):
            self.print_pruning_stats(runner.model)

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        self.save_history(runner)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        pass
