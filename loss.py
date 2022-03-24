import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ReWeighting(nn.Module):
    def __init__(self, cls_num_list=None):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.m_list = cls_num_list
            train_size_arr = np.array(self.m_list)
            train_size_mean = np.mean(train_size_arr)
            train_size_factor = train_size_mean / train_size_arr
            self.per_cls_weights = torch.from_numpy(train_size_factor).type(torch.FloatTensor)

    def to(self, device):
        super().to(device)

        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)

        return self

    def forward(self, output_logits, target):
        if self.m_list:
            return F.cross_entropy(output_logits,target,weight=self.per_cls_weights,reduction='mean')
        else:
            return F.cross_entropy(output_logits, target)


class LogitAdjustLoss(nn.Module):
    """
    Paper: Long-Tail Learning via Logit Adjustment (ICLR 2021)
    arXiv: https://arxiv.org/abs/2007.07314
    Source Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    """
    def __init__(self, cls_num_list=None, reweight_epoch=-1, tau=1):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = torch.from_numpy(np.array(cls_num_list))
            self.m_list = m_list / m_list.sum()
            self.tau = tau

            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        adjust_term = (self.m_list.unsqueeze(0) + 1e-12).log() * self.tau
        adjust_term = adjust_term.detach()

        final_output = x + adjust_term

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class ALALoss(nn.Module):
    """
    Paper: Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition (AAAI 2022)
    arXiv: https://arxiv.org/abs/2104.06094
    """
    def __init__(self, cls_num_list=None, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = torch.from_numpy(np.array(cls_num_list))
            m_list = np.log(2) / torch.log(m_list / m_list.min() + 1)
            # m_list = 1.0 / torch.log(m_list / m_list.min() + 1)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):

        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        hardness_term = output_logits / self.s
        hardness_term = (1 - hardness_term) / 2.0 * index
        hardness_term = hardness_term.detach()

        adjust_term = self.m_list[target].unsqueeze(-1)
        adjust_term = adjust_term * index
        adjust_term = adjust_term.detach()

        final_output = x - hardness_term * adjust_term * self.s

        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)