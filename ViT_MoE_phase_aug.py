"""
ViT_MoE_phase_aug.py
MoE-FFD + Phase-Only Reconstruction Module

Architecture:
  - Backbone  : ViT-B/16 (ImageNet-21k pretrained, frozen)
  - LoRA MoE  : Mixture-of-Experts low-rank adaptation on attention QKV
  - Adapter MoE: Conv2d-Diff adapter experts on FFN (cd/ad/rd/scd/cv kernels)
  - PhaseModule: FFT phase extraction -> lightweight CNN -> routing feature z_phase

Key idea:
  JPEG compression destroys magnitude but preserves phase.
  PhaseModule feeds z_phase into MoE routing gates so the model
  learns compression-invariant expert selection.

Reference: based on MoE-FFD (https://arxiv.org/abs/2408.xxxxx)
"""

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
from torch.distributions.normal import Normal


_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Conv2d-Diff operators (cv / cd / ad / rd / scd)
# ──────────────────────────────────────────────────────────────

def createConvFunc(op_type):
    assert op_type in ['cv', 'cd', 'ad', 'rd', 'scd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2]
            assert weights.size(2) == 3 and weights.size(3) == 3
            assert padding == dilation
            weights_c = weights.sum(dim=[2, 3]) - weights[:,:,1,1]
            weights_c = weights_c[:,:,None,None]
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2]
            assert weights.size(2) == 3 and weights.size(3) == 3
            assert padding == dilation
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_c = weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
            weights_c[:,:,4] = weights[:,:,4]*0
            weights_conv = (weights - weights_c).view(shape)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2]
            assert weights.size(2) == 3 and weights.size(3) == 3
            padding = 2 * dilation
            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = weights[:, :, 0]
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'scd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2]
            assert weights.size(2) == 3 and weights.size(3) == 3
            padding = 2 * dilation
            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:] * 2
            buffer[:, :, 12] = weights.sum(dim=[2])
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func


class Conv2d_Diff(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, op_type='cv'):
        super().__init__()
        assert op_type in ['cv', 'cd', 'ad', 'rd', 'scd']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.func = createConvFunc(op_type)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Conv2d_Adapter(nn.Module):
    def __init__(self, dim, adapter_dim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, op_type='cv'):
        super().__init__()
        self.adapter_down = nn.Linear(dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.adapter_dim = adapter_dim
        self.adapter_conv = Conv2d_Diff(adapter_dim, adapter_dim, kernel_size,
                                        stride, padding, dilation, groups, bias, op_type)
        nn.init.zeros_(self.adapter_conv.weight)
        self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(adapter_dim, dtype=torch.float)

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.adapter_dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.adapter_dim)
        x_cls = x_down[:, :1].reshape(B, 1, 1, self.adapter_dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.adapter_dim)
        x_down = torch.cat([x_cls, x_patch], dim=1)
        x_up = self.adapter_up(x_down)
        return x_up


# ──────────────────────────────────────────────────────────────
# Phase-Only Reconstruction Module
# ──────────────────────────────────────────────────────────────

class PhaseModule(nn.Module):
    """
    Extracts phase-only features for compression-robust MoE routing.

    Steps:
      1. 2D FFT on input image
      2. Set magnitude=1, keep only phase angle  (phase-only reconstruction)
      3. Inverse FFT -> phase image in spatial domain
      4. Lightweight CNN -> z_phase [B, embed_dim] used as routing signal
    """
    def __init__(self, in_chans=3, feat_dim=128, embed_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, feat_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(feat_dim, embed_dim)

    def phase_reconstruct(self, x):
        fft = torch.fft.fft2(x, norm='ortho')
        phase = torch.angle(fft)
        phase_fft = torch.exp(1j * phase)
        phase_img = torch.fft.ifft2(phase_fft, norm='ortho').real
        phase_img = phase_img - phase_img.mean(dim=(-2, -1), keepdim=True)
        std = phase_img.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        phase_img = phase_img / std
        return phase_img

    def forward(self, x):
        phase_img = self.phase_reconstruct(x)
        feat = self.encoder(phase_img).flatten(1)
        z_phase = self.proj(feat)
        return z_phase, phase_img


# ──────────────────────────────────────────────────────────────
# MoE layers
# ──────────────────────────────────────────────────────────────

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=False):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Adapter_MoElayer(nn.Module):
    def __init__(self, dim=768, adapter_dim=8,
                 adapter_type=['cv', 'cd', 'ad', 'rd', 'scd'],
                 noisy_gating=True, k=1,
                 freq_dim=0, uncertainty_routing=False, entropy_thresh=1.0):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.k = k
        self.identity = nn.Identity()
        self.freq_dim = freq_dim
        self.uncertainty_routing = uncertainty_routing
        self.entropy_thresh = entropy_thresh

        adapter_experts = nn.ModuleList()
        for t in adapter_type:
            adapter_experts.append(Conv2d_Adapter(dim=dim, adapter_dim=adapter_dim,
                                                   kernel_size=3, stride=1, padding=1,
                                                   bias=True, op_type=t))
        self.num_experts = len(adapter_experts)
        self.adapter_experts = adapter_experts
        gate_in_dim = dim + freq_dim
        self.w_gate = nn.Parameter(torch.zeros(gate_in_dim, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(gate_in_dim, self.num_experts), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.k_max = max(self.k, 2) if self.uncertainty_routing else self.k
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        full_probs = self.softmax(clean_logits)
        entropy = -(full_probs * torch.log(full_probs + 1e-10)).sum(dim=1)

        k_max = self.k_max
        top_logits, top_indices = logits.topk(min(k_max + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :k_max]
        top_k_indices = top_indices[:, :k_max]
        top_k_gates = self.softmax(top_k_logits)

        if self.uncertainty_routing and k_max > 1:
            slot_idx = torch.arange(k_max, device=x.device).unsqueeze(0)
            keep_k = torch.where(entropy < self.entropy_thresh,
                                  torch.ones_like(entropy),
                                  torch.full_like(entropy, float(k_max)))
            slot_mask = (slot_idx < keep_k.unsqueeze(1)).float()
            top_k_gates = top_k_gates * slot_mask
            top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-10)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k_max < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, entropy

    def forward(self, x, z_freq=None, loss_coef=1):
        B, N, _ = x.shape
        x_global = torch.mean(x, dim=1, keepdim=False)
        gate_input = torch.cat([x_global, z_freq], dim=-1) if (self.freq_dim > 0 and z_freq is not None) else x_global

        gates, load, entropy = self.noisy_top_k_gating(gate_input, self.training)
        importance = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            expert_output = self.adapter_experts[i](expert_inputs[i])
            expert_output = expert_output.reshape(expert_output.size(0), 197 * self.dim)
            expert_outputs.append(expert_output)

        y = dispatcher.combine(expert_outputs)
        y = y.reshape(B, 197, self.dim)
        return y, loss, entropy


class LoRA_MoElayer(nn.Module):
    def __init__(self, dim, lora_dim=[8, 16, 32, 48, 64, 96, 128], noisy_gating=True, k=1):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.k = k

        Lora_a_experts = nn.ModuleList()
        Lora_b_experts = nn.ModuleList()
        for i, d in enumerate(lora_dim):
            Lora_a_experts.append(nn.Linear(dim, d, bias=False))
            nn.init.kaiming_uniform_(Lora_a_experts[i].weight, a=math.sqrt(5))
            Lora_b_experts.append(nn.Linear(d, dim * 3, bias=False))
            nn.init.zeros_(Lora_b_experts[i].weight)

        self.num_experts = len(Lora_a_experts)
        self.Lora_a_experts = Lora_a_experts
        self.Lora_b_experts = Lora_b_experts
        self.w_gate = nn.Parameter(torch.zeros(dim, len(Lora_a_experts)), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(dim, len(Lora_a_experts)), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        B, N, C = x.shape
        x = x.reshape(B * N, C)
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()

        expert_outputs = []
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            qkv_delta = F.linear(expert_inputs[i], self.Lora_a_experts[i].weight)
            qkv_delta = F.linear(qkv_delta, self.Lora_b_experts[i].weight)
            expert_outputs.append(qkv_delta)
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(B, N, C * 3)
        return y, loss


# ──────────────────────────────────────────────────────────────
# ViT config
# ──────────────────────────────────────────────────────────────

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
}


# ──────────────────────────────────────────────────────────────
# ViT blocks
# ──────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., lora_topk=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.LoRA_k = lora_topk
        if self.LoRA_k > 0:
            self.LoRA_MoE = LoRA_MoElayer(dim, k=self.LoRA_k)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.LoRA_k > 0:
            qkv_delta, lora_loss = self.LoRA_MoE(x)
            qkv_delta = qkv_delta.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_delta, k_delta, v_delta = qkv_delta.unbind(0)
            q, k, v = q + q_delta, k + k_delta, v + v_delta
        else:
            lora_loss = torch.zeros(1).to(x.device)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, lora_loss


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 lora_topk=1, adapter_topk=1, freq_dim=0,
                 uncertainty_routing=False, entropy_thresh=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, lora_topk=lora_topk)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.adapter_k = adapter_topk
        if self.adapter_k > 0:
            self.adapter_MoE = Adapter_MoElayer(dim, adapter_dim=8, k=self.adapter_k,
                                                 freq_dim=freq_dim,
                                                 uncertainty_routing=uncertainty_routing,
                                                 entropy_thresh=entropy_thresh)

    def forward(self, x, z_freq=None):
        x1, lora_loss = self.attn(self.norm1(x))
        x = x + self.drop_path(x1)
        if self.adapter_k > 0:
            x_adapter, adapter_loss, entropy = self.adapter_MoE(self.norm2(x), z_freq=z_freq)
            x_adapter = self.drop_path(x_adapter)
            x = x + x_adapter + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            adapter_loss = torch.zeros(1).to(x.device)
            entropy = None
        return x, lora_loss, adapter_loss, entropy


# ──────────────────────────────────────────────────────────────
# VisionTransformer
# ──────────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 weight_init='', lora_topk=1, adapter_topk=1,
                 use_freq_module=False, freq_feat_dim=128, freq_band_mode='all',
                 uncertainty_routing=False, entropy_thresh=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.use_freq_module = use_freq_module
        freq_dim_for_gate = embed_dim if use_freq_module else 0
        if use_freq_module:
            self.freq_module = PhaseModule(in_chans=in_chans, feat_dim=freq_feat_dim, embed_dim=embed_dim)
        else:
            self.freq_module = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  lora_topk=lora_topk, adapter_topk=adapter_topk,
                  freq_dim=freq_dim_for_gate,
                  uncertainty_routing=uncertainty_routing, entropy_thresh=entropy_thresh)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.lora_topk = lora_topk
        self.adapter_topk = adapter_topk
        self.uncertainty_routing = uncertainty_routing
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Auxiliary phase head for consistency loss (training only)
        self.phase_head = nn.Linear(self.num_features, num_classes) if (num_classes > 0 and use_freq_module) else None

        self.freeze_stages()

    def freeze_stages(self):
        self.pos_drop.eval()
        self.patch_embed.eval()
        for block in self.blocks:
            block.eval()
            if self.lora_topk > 0:
                block.attn.LoRA_MoE.train()
            if self.adapter_topk > 0:
                block.adapter_MoE.train()

        for name, param in self.named_parameters():
            if ('LoRA' not in name and 'adapter' not in name and 'head' not in name
                    and 'norm1' not in name and 'freq_module' not in name):
                param.requires_grad = False

        total = lora_n = adapter_n = head_n = phase_n = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.numel()
                if 'LoRA' in name:      lora_n += param.numel()
                elif 'head' in name:    head_n += param.numel()
                elif 'adapter' in name: adapter_n += param.numel()
                elif 'freq_module' in name: phase_n += param.numel()
        print(f'parameters: {total}  LoRA={lora_n}  adapter={adapter_n}  head={head_n}  phase_module={phase_n}')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x_raw):
        if self.use_freq_module:
            z_freq, phase_img = self.freq_module(x_raw)
        else:
            z_freq = None
            phase_img = None

        x = self.patch_embed(x_raw)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        lora_loss_list, adapter_loss_list, entropy_list = [], [], []
        for block in self.blocks:
            x, cur_lora_loss, cur_adapter_loss, cur_entropy = block(x, z_freq=z_freq)
            lora_loss_list.append(cur_lora_loss)
            adapter_loss_list.append(cur_adapter_loss)
            if cur_entropy is not None:
                entropy_list.append(cur_entropy)

        lora_loss = torch.mean(torch.stack(lora_loss_list))
        adapter_loss = torch.mean(torch.stack(adapter_loss_list))
        moe_loss = lora_loss * 200 + adapter_loss * 1

        x = self.norm(x)
        cls_feat = x[:, 0]
        entropy_mean = torch.stack(entropy_list).mean(dim=0) if entropy_list else None

        return self.pre_logits(cls_feat), moe_loss, phase_img, entropy_mean, z_freq

    def forward(self, x):
        cls_feat, moe_loss, phase_img, entropy_mean, z_freq = self.forward_features(x)
        logits = self.head(cls_feat)

        consistency_loss = torch.tensor(0.0, device=x.device)
        logits_phase = None

        if self.phase_head is not None and z_freq is not None and self.training:
            T = 2.0
            logits_phase = self.phase_head(z_freq.detach())
            p_phase = F.softmax(logits_phase.detach() / T, dim=-1)
            p_spatial = F.log_softmax(logits / T, dim=-1)
            consistency_loss = F.kl_div(p_spatial, p_phase, reduction='batchmean') * (T ** 2)

        return logits, moe_loss, phase_img, entropy_mean, consistency_loss, logits_phase


# ──────────────────────────────────────────────────────────────
# Weight init / loading helpers
# ──────────────────────────────────────────────────────────────

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if gs_new:
        gs_h, gs_w = gs_new
    else:
        gs_h = gs_w = int(math.sqrt(posemb_new.shape[1] - num_tokens))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:   w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3: w = w.transpose([2, 0, 1])
            elif w.ndim == 2: w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    embed_conv_w = adapt_input_conv(
        model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed,
                                       getattr(model, 'num_tokens', 1),
                                       model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.blocks):
        bp = f'{prefix}Transformer/encoderblock_{i}/'
        block.norm1.weight.copy_(_n2p(w[f'{bp}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{bp}LayerNorm_0/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{bp}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{bp}LayerNorm_2/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{bp}MultiHeadDotProductAttention_1/{n}/kernel'], t=False).flatten(1).T
            for n in ['query', 'key', 'value']]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{bp}MultiHeadDotProductAttention_1/{n}/bias'], t=False).reshape(-1)
            for n in ['query', 'key', 'value']]))
        block.attn.proj.weight.copy_(
            _n2p(w[f'{bp}MultiHeadDotProductAttention_1/out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(
            _n2p(w[f'{bp}MultiHeadDotProductAttention_1/out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{bp}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{bp}MlpBlock_3/Dense_{r}/bias']))


def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        representation_size=None, **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224_in21k']
    if pretrained:
        import timm
        pretrained_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('Loaded pretrained ViT-B/16 (ImageNet-21k) weights.')
    return model
