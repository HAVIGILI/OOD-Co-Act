# Detector class
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from mmseg.apis import inference_segmentor
from typing import Tuple
from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score,
                             precision_recall_curve)
# För TorchMetrics-baserade metoder
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryROC
import os
from typing import List
from matplotlib.colors import ListedColormap

#########################################################
#  Inference wrapper that applies LINe-style masking    #
#  + four pairwise-co-activation variants               #
#########################################################
class AnomalyDetector:
    def __init__(self,
                 anomaly_images,
                 anomaly_IDs,
                 model,
                 state_dict):
        # ----- unpack model & LINe state -----
        self.model                         = model
        self.device = next(model.parameters()).device
        self.contribution_tensor           = state_dict['contrib'].to(self.device)
        self.average_activations_per_class = state_dict['avg_act'].to(self.device)

        # ----- thresholds -----
        self.activation_clipping     = 1000
        self.activation_pruning      = 0
        self.weight_pruning          = 0
        self.temperature_co          = 1
        self.temperature_line        = 1
        self.inverse_convert_to_ones = False
        self.wt_threshold            = 0
        self.wt_u_threshold          = 1
        self.co_weighted             = 1
        self.max_pool                = 0
        self.use_elsewhere_map       = None
        self.weight_mode             = "softmax"
        self.make_feat_binary        = False
        self.c_u_ratio               = False

        # ----- unpack coact stats -----
        self.mean_freq     = state_dict['freq_bin'].to(self.device)      # P[aᵢ>0 ∧ aⱼ>0]     freq_bin is not binary.
        self.mean_wt       = state_dict['mean_wt'].to(self.device)      # E[aᵢ·aⱼ]
        self.mean_wt_both  = state_dict['mean_wt_both'].to(self.device)  # E[aᵢ·aⱼ | aᵢ>0∧aⱼ>0]
        self.mean_wt_any   = state_dict['mean_wt_any'].to(self.device)   # E[aᵢ·aⱼ | aᵢ>0∨aⱼ>0]

        # These are not used, yet. They are not class-specific. Any needs an "all" as well
        self.freq_bin_all     = state_dict['freq_bin'].to(self.device)
        self.mean_wt_all       = state_dict['mean_wt'].to(self.device)
        self.mean_wt_all       = state_dict['mean_wt_both'].to(self.device)

        self.common_freq   = None
        self.common_wt     = None
        self.common_both   = None
        self.common_any    = None
        self.uncommon_freq = None
        self.uncommon_wt   = None
        self.uncommon_both = None
        self.uncommon_any  = None

        # ----- prepare masks for LINe -----
        self.activation_mask     = torch.zeros(19, 512, device=self.device)
        self.common_neurons_mask = torch.zeros(19, 512, device=self.device)
        self.weight_mask         = torch.zeros(19, 19, 512, device=self.device)

        self.model_output      = None
        self.line_model_output = [0] * 19
        self.model_IDs_output  = None
        self.nr_of_activations = None
        self.input             = None
        self.feature_map       = None

        # ----- data & plotting -----
        self.anomaly_images = anomaly_images
        self.anomaly_IDs    = anomaly_IDs
        self.images_to_be_plotted = []
        self.plot_pixel_activations = False
        self.plot_pair_ratios = False
        

        # ----- fixed conv weight tensor -----
        self.weight_tensor = (self.model.decode_head.conv_seg.weight
                              .data.squeeze(-1).squeeze(-1))

        self.setup_masks()
        self.id_dot = (10, 10)
        self.ood_dot = (10, 10)

        # ----- maps containers -----
        self.gt_maps            = []
        self.raw_line_maps      = []
        self.raw_softmax_maps   = []
        self.raw_logitmin_maps  = []
        self.raw_logitmax_maps  = []
        self.raw_nract_maps     = []
        self.raw_euclidian_maps = []
        self.raw_coact_bin      = []
        self.raw_coact_wt       = []
        self.raw_coact_both     = []
        self.raw_coact_any      = []

        self.co_blur_ksize      = 0
        self.clips              = (0,0)
        self.co_sigma           = 0
        self.baseline_and_line_blur_ksize = (0,0)
        self.baseline_and_line_sigma = 0

        # ----- register hook on conv_seg -----
        self.model.decode_head.conv_seg._forward_hooks.clear()
        self.model.decode_head.conv_seg.register_forward_hook(
            self.hook
        )

    def hook(self, module, inp, out):
        line_input = inp[0].clone()
        self.input       = line_input
        self.feature_map = line_input.detach()

        # standard conv
        pred_logits = F.conv2d(line_input,
                                module.weight,
                                module.bias,
                                stride=module.stride,
                                padding=module.padding,
                                dilation=module.dilation,
                                groups=module.groups)
        pred_ids = torch.argmax(pred_logits, dim=1)

        # LINe per class (masking + conv). It is possible to remove bias
        for cls in range(19):
            mask_c = self.activation_mask[cls].view(1,512,1,1)
            in2 = line_input.clamp(max=self.activation_clipping) * mask_c
            w_m = self.weight_mask[cls].view_as(module.weight)
            upd_w = module.weight * w_m
            self.line_model_output[cls] = F.conv2d(
                in2, upd_w, module.bias,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups
            )

        self.nr_of_activations = (line_input.squeeze(0)>0).sum(dim=0)
        self.model_output      = pred_logits
        self.model_IDs_output  = pred_ids
        return out

    def ood_inference(self):
        """
        Run inference, and get raw maps for all methods.
        """
        self.setup_masks()
        for i, (img_p, gt_p) in enumerate(zip(self.anomaly_images, self.anomaly_IDs)):
            result = inference_segmentor(self.model, img_p)
            gt     = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)

            nr_act    = self._calculate_nr_activation_scores(gt)
            line      = self._calculate_line_scores(gt)
            softmax   = self._calculate_softmax_scores(gt)
            logitmin  = self._calculate_logitmin_scores(gt)
            logitmax  = self._calculate_logitmax_scores(gt)
            euclidian = self._calculate_euclidean_scores(gt)
            coact_any  = self._calculate_coactivation_score(gt, self.common_any, self.uncommon_any)
            coact_bin  =  self._calculate_coactivation_score(gt, self.common_freq, self.uncommon_freq)
            coact_wt   =  self._calculate_coactivation_score(gt, self.common_wt, self.uncommon_wt)
            coact_both = self._calculate_coactivation_score(gt, self.common_both, self.uncommon_both)

            # six original maps
            self.raw_nract_maps.append(     nr_act)
            self.raw_line_maps.append(      line)
            self.raw_softmax_maps.append(   softmax)
            self.raw_logitmin_maps.append(  logitmin)
            self.raw_logitmax_maps.append(  logitmax)
            self.raw_euclidian_maps.append( euclidian)

            # four coact variants
            self.raw_coact_bin.append(      coact_bin)
            self.raw_coact_wt.append(       coact_wt)
            self.raw_coact_both.append(     coact_both)
            self.raw_coact_any.append(      coact_any)

            self.gt_maps.append(gt)
            if i in self.images_to_be_plotted:
                self.plot_maps(result, img_p, gt,
                              self.raw_line_maps[-1],
                              self.raw_softmax_maps[-1],
                              self.raw_logitmin_maps[-1],
                              self.raw_logitmax_maps[-1],
                              self.raw_euclidian_maps[-1],
                              self.raw_nract_maps[-1],
                              self.raw_coact_bin[-1],
                              self.raw_coact_wt[-1],
                              self.raw_coact_both[-1],
                              self.raw_coact_any[-1])

    def get_ood_score_lists(self):
        # flatten GT and filter ignore
        all_gt = np.concatenate([m.ravel() for m in self.gt_maps])
        valid  = (all_gt != 255)

        # collect all ten methods
        scores = {
            "ood_scores_line":          np.concatenate([m.ravel() for m in self.raw_line_maps])[valid],
            "ood_scores_softmax":       np.concatenate([m.ravel() for m in self.raw_softmax_maps])[valid],
            "ood_scores_logitmin":      np.concatenate([m.ravel() for m in self.raw_logitmin_maps])[valid],
            "ood_scores_logitmax":      np.concatenate([m.ravel() for m in self.raw_logitmax_maps])[valid],
            "ood_scores_nr_activation": np.concatenate([m.ravel() for m in self.raw_nract_maps])[valid],
            "ood_scores_euclidian":     np.concatenate([m.ravel() for m in self.raw_euclidian_maps])[valid],
            "ood_scores_coact_bin":     np.concatenate([m.ravel() for m in self.raw_coact_bin])[valid],
            "ood_scores_coact_wt":      np.concatenate([m.ravel() for m in self.raw_coact_wt])[valid],
            "ood_scores_coact_both":    np.concatenate([m.ravel() for m in self.raw_coact_both])[valid],
            "ood_scores_coact_any":     np.concatenate([m.ravel() for m in self.raw_coact_any])[valid],
        }
        return all_gt[valid], scores

    # --------------------------------------------------------------
    #  Score-map calculators
    # --------------------------------------------------------------
    def _calculate_line_scores(self, labels):
        ood = np.zeros((256, 512))
        for ID in range(19):
            m = (self.model_IDs_output == ID).squeeze(0).cpu().numpy()
            logits = self.line_model_output[ID].squeeze(0)
            ood[m] = -(torch.logsumexp(logits / self.temperature_line, dim=0) * self.temperature_line).cpu().numpy()[m]

        self.use_elsewhere_map = ood #This is done to combine methods first, and do blurring as well, but after.
        ood = cv2.resize(ood, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(ood, self.baseline_and_line_blur_ksize, self.baseline_and_line_sigma) if self.baseline_and_line_blur_ksize != (0, 0) else ood

    def _calculate_logitmax_scores(self, labels):
        ood = -torch.max(self.model_output, dim=1).values.squeeze(0).cpu().numpy()
        ood = cv2.resize(ood, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(ood, self.baseline_and_line_blur_ksize, self.baseline_and_line_sigma) if self.baseline_and_line_blur_ksize != (0, 0) else ood

    def _calculate_softmax_scores(self, labels):
        sm  = F.softmax(self.model_output, dim=1)
        ood = -torch.max(sm, dim=1).values.squeeze(0).cpu().numpy()
        ood = cv2.resize(ood, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(ood, self.baseline_and_line_blur_ksize, self.baseline_and_line_sigma) if self.baseline_and_line_blur_ksize != (0, 0) else ood

    def _calculate_logitmin_scores(self, labels):
        ood = -torch.min(self.model_output, dim=1).values.squeeze(0).cpu().numpy()
        ood = cv2.resize(ood, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(ood, self.baseline_and_line_blur_ksize, self.baseline_and_line_sigma) if self.baseline_and_line_blur_ksize != (0, 0) else ood

    def _calculate_nr_activation_scores(self, labels):
        ood = np.zeros((256, 512))
        for ID in range(19):
            act_imp = ((self.input > 0) *
                       self.common_neurons_mask[ID].view(1, 512, 1, 1))
            m = (self.model_IDs_output == ID).squeeze(0).cpu().numpy()
            cur = act_imp.sum(dim=1).squeeze(0).cpu().numpy()
            ood[m] = cur[m]
        #self.use_elsewhere_map = ood
        ood = cv2.resize(ood, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(ood, self.baseline_and_line_blur_ksize, self.baseline_and_line_sigma) if self.baseline_and_line_blur_ksize != (0, 0) else ood

    def _calculate_euclidean_scores(self, labels, eps=1e-9):
        _, _, Hf, Wf = self.input.shape
        H_img, W_img = labels.shape
        dev          = self.input.device

        avg = (self.contribution_tensor /
                (self.weight_tensor.abs() + eps)).to(dev)

        feat  = self.input.squeeze(0).permute(1,2,0).reshape(-1, 512)
        cls   = self.model_IDs_output.squeeze(0).reshape(-1)
        dist  = (feat - avg[cls]).pow(2).sum(1).sqrt()

        ood = dist.view(Hf, Wf).cpu().numpy()

        # If using lowest distance instead of prediction:

        # _, _, Hf, Wf = self.input.shape
        # H_img, W_img = labels.shape
        # dev          = self.input.device

        # # avg: [K, C], K = number of classes (e.g. 19), C = feature dim (512)
        # avg = (self.contribution_tensor / (self.weight_tensor.abs() + eps)).to(dev)

        # # 1) flatten feature map → [N, C]
        # feat = (
        #     self.input
        #         .squeeze(0)                    # [C, Hf, Wf]
        #         .permute(1, 2, 0)              # [Hf, Wf, C]
        #         .reshape(-1, 512)              # [N, C]
        #         .to(dev)
        # )

        # # 2) compute squared distances to every class-mean → [N, K]
        # #    feat.unsqueeze(1): [N, 1, C], avg.unsqueeze(0): [1, K, C]
        # dists_sq = (feat.unsqueeze(1) - avg.unsqueeze(0)).pow(2).sum(dim=2)  # [N, K]

        # # 3) take sqrt of the minimum over classes → [N]
        # min_dist = dists_sq.min(dim=1).values.sqrt()  # [N]

        # # 4) reshape back to [Hf, Wf], resize, and blur
        # ood = min_dist.view(Hf, Wf).cpu().numpy()

        ood = cv2.resize(ood, labels.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(ood, self.baseline_and_line_blur_ksize, self.baseline_and_line_sigma) if self.baseline_and_line_blur_ksize != (0, 0) else ood

    def setup_masks(self):
        # initialize accumulators for per-class weighting
        weighted_freq  = torch.zeros_like(self.mean_freq)
        weighted_wt    = torch.zeros_like(self.mean_wt)
        weighted_both  = torch.zeros_like(self.mean_wt_both)
        weighted_any   = torch.zeros_like(self.mean_wt_any)


        # Weight matrix used for experimenting with weights for the Co-act matrix. Can be ignored
        weight_co_tensor = torch.zeros_like(self.mean_freq)
        for ID in range(19):
            w_raw = self.weight_tensor[ID].view(-1,1)
            w_pos = torch.relu(w_raw)          # positives
            w_neg = torch.relu(-w_raw)         # negatives as positives
            w_abs = w_raw.abs()                # magnitude
            w_raw = self.weight_tensor[ID].view(-1,1)
            pos   = (w_raw > 0).float()   # 1 where wᵢ>0
            neg   = (w_raw < 0).float()   # 1 where wᵢ<0
            # 1) only positive–positive pairs
            mat_pospos = pos @ pos.T        # ones only where wᵢ>0 AND wⱼ>0

            # 2) only negative–negative pairs
            mat_negneg = neg @ neg.T        # ones only where wᵢ<0 AND wⱼ<0

            # 3) only negative–positive (i negative, j positive)
            mat_negpos = neg @ pos.T        # ones only where wᵢ<0 AND wⱼ>0

            # (optionally the flip: positive–negative)
            mat_posneg = pos @ neg.T        # ones only where wᵢ>0 AND wⱼ<0


            if   self.co_weighted ==  0:
              weight_matrix = torch.ones_like(self.mean_freq[ID])
            elif self.co_weighted ==  1:
                weight_matrix = w_pos @ w_pos.T
            elif self.co_weighted ==  2:
                weight_matrix = w_pos + w_pos.T
            elif self.co_weighted ==  3:
                weight_matrix = w_neg @ w_neg.T
            elif self.co_weighted ==  4:
                weight_matrix = w_neg + w_neg.T
            elif self.co_weighted ==  5:
                weight_matrix = w_raw @ w_raw.T
            elif self.co_weighted ==  6:
                weight_matrix = -(w_raw + w_raw.T)
            elif self.co_weighted ==  7:
                weight_matrix = w_abs @ w_abs.T
            elif self.co_weighted ==  8:
                weight_matrix = w_abs + w_abs.T
            elif self.co_weighted ==  9:
                weight_matrix = w_pos + w_neg.T
            elif self.co_weighted == 10:
                weight_matrix = w_neg + w_pos.T
            elif self.co_weighted == 11:
                weight_matrix = mat_pospos
            elif self.co_weighted == 12:
                weight_matrix = mat_negneg
            elif self.co_weighted == 13:
                weight_matrix = mat_negpos
            elif self.co_weighted == 14:
                weight_matrix = mat_posneg
            elif   self.co_weighted == 15:
                # broadcast: [-w_neg] is [C,1], w_pos.T is [1,C] → result [C,C]
                weight_matrix = -(-w_neg + w_pos.T)
            elif self.co_weighted == 16:
                # sum of absolute weights, but keep the original sign of (w_i + w_j)
                w_sum = w_raw + w_raw.T                # [C,C]
                m     = w_abs + w_abs.T                # [C,C]
                weight_matrix = -m * torch.sign(w_sum) # [C,C]
            elif self.co_weighted == 17:
                weight_matrix = w_neg @ w_pos.T
            else:
                raise ValueError(f"Unknown co_weighted={self.co_weighted}")

            weight_co_tensor[ID] = weight_matrix
            weighted_freq[ID] = self.mean_freq[ID]
            weighted_wt[ID] = self.mean_wt[ID]
            weighted_both[ID] = self.mean_wt_both[ID]
            weighted_any[ID] = self.mean_wt_any[ID]

        # Build binary masks
        self.common_freq_mask = (weighted_freq  >= torch.quantile(weighted_freq.view(19, -1),
                                                                  self.wt_threshold, dim=1).view(19, 1, 1)).float()
        self.common_wt_mask = (weighted_wt  >= torch.quantile(weighted_wt.view(19, -1),
                                                              self.wt_threshold, dim=1).view(19, 1, 1)).float()
        self.common_both_mask = (weighted_both >= torch.quantile(weighted_both.view(19, -1),
                                                                self.wt_threshold, dim=1).view(19, 1, 1)).float()
        self.common_any_mask = (weighted_any >= torch.quantile(weighted_any.view(19, -1),
                                                              self.wt_threshold, dim=1).view(19, 1, 1)).float()

        self.uncommon_freq_mask = (weighted_freq <= torch.quantile(weighted_freq.view(19, -1),
                                                                  self.wt_u_threshold, dim=1).view(19, 1, 1)).float()
        self.uncommon_wt_mask = (weighted_wt <= torch.quantile(weighted_wt.view(19, -1),
                                                              self.wt_u_threshold, dim=1).view(19, 1, 1)).float()
        self.uncommon_both_mask = (weighted_both <= torch.quantile(weighted_both.view(19, -1),
                                                                  self.wt_u_threshold, dim=1).view(19, 1, 1)).float()
        self.uncommon_any_mask = (weighted_any <= torch.quantile(weighted_any.view(19, -1),
                                                                self.wt_u_threshold, dim=1).view(19, 1, 1)).float()
        # Create inverses
        eps = 1e-12
        self.common_freq = self.common_freq_mask / (weighted_freq + eps) * weight_co_tensor
        self.common_wt = self.common_wt_mask / (weighted_wt + eps) * weight_co_tensor
        self.common_both = self.common_both_mask / (weighted_both + eps) * weight_co_tensor
        self.common_any = self.common_any_mask / (weighted_any + eps) * weight_co_tensor

        self.uncommon_freq = self.uncommon_freq_mask / (weighted_freq + eps) * weight_co_tensor
        self.uncommon_wt = self.uncommon_wt_mask / (weighted_wt + eps) * weight_co_tensor
        self.uncommon_both = self.uncommon_both_mask / (weighted_both + eps) * weight_co_tensor
        self.uncommon_any = self.uncommon_any_mask / (weighted_any + eps) * weight_co_tensor

        # Create LINe masks
        for ID in range(19):
            thr = torch.quantile(self.contribution_tensor[ID], self.activation_pruning)
            self.activation_mask[ID] = (self.contribution_tensor[ID] >= thr).float()
            self.weight_mask[ID] = self.weight_tensor * self.contribution_tensor[ID]
            thr_w = torch.quantile(self.weight_mask[ID], self.weight_pruning)
            self.weight_mask[ID] = (self.weight_mask[ID] >= thr_w).float()
            thr_c = torch.quantile(self.average_activations_per_class[ID], 0)                        # To include common neurons instead of combinations, can be ignored
            self.common_neurons_mask[ID] = (self.average_activations_per_class[ID] > thr_c).float()  # To include common neurons instead of combinations, can be ignored

    # --------------------------------------------------------------
    #  Visualisation
    # --------------------------------------------------------------
    def plot_maps(self, result, img_path, gt, line_map, softmax_map, logitmin_map, logitmax_map, euclidian_map, nr_act_map, coact_map1, coact_map2, coact_map3, coact_map4):
        seg_map = result[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H_img, W_img = gt.shape

        dots = [self.ood_dot, self.id_dot]

        # To plot the penultimate activations. Modify the function however you want to plot it (for example uncommon to the left, or low contributing to the left)
        if self.plot_pixel_activations:
            self.plot_pixel_activation((self.ood_dot[0], self.ood_dot[1]), kind="OOD",    img_shape=(H_img, W_img))
            self.plot_pixel_activation((self.id_dot[0], self.ood_dot[1]),  kind="Normal", img_shape=(H_img, W_img))
        if self.plot_pair_ratios:
            self.plot_pair_ratio(self.ood_dot, (H_img, W_img), nbins=50)
            self.plot_pair_ratio(self.id_dot,  (H_img, W_img), nbins=50)


        # Show all heatmaps
        print("nr activations:")
        self.plotter(img, seg_map, nr_act_map, gt, dots)
        print("line:")
        self.plotter(img, seg_map, line_map, gt, dots)
        print("softmax:")
        self.plotter(img, seg_map, softmax_map, gt, dots)
        print("logitmin:")
        self.plotter(img, seg_map, logitmin_map, gt, dots)
        print("logitmax:")
        self.plotter(img, seg_map, logitmax_map, gt, dots)
        print("euclidian:")
        self.plotter(img, seg_map, euclidian_map, gt, dots)
        print("coact_bin:")
        self.plotter(img, seg_map, coact_map1, gt, dots)
        print("coact_wt:")
        self.plotter(img, seg_map, coact_map2, gt, dots)
        print("coact_both:")
        self.plotter(img, seg_map, coact_map3, gt, dots)
        print("coact_any:")
        self.plotter(img, seg_map, coact_map4, gt, dots)

    def plotter(self,
                            img: np.ndarray,
                            seg_map: np.ndarray,
                            ood_map: np.ndarray,
                            gt: np.ndarray,
                            dots: List[Tuple[int,int,str]] = None):
        """
        Creates a 2×2 figure:
          [ (1,1) RGB + seg overlay       | (1,2) raw heatmap             ]
          [ (2,1) masked heatmap (ignore)$ | (2,2) ground‐truth (ignore)$  ]

        Arguments:
            img     : H×W×3 RGB image (uint8)
            seg_map : H×W segmentation map in [0..18] that you overlay with alpha
            ood_map : H×W float array (raw OOD‐scores)
            gt      : H×W integer ground truth with {0,1,…,18, 255=ignore}
            dots    : optional list of (y, x, color) to scatter
        """

        ignore_label = 255

        # Precompute vmin/vmax for the raw heatmap
        vmin, vmax = float(ood_map.min()), float(ood_map.max())

        # Build the masked‐OOD map (ignore pixels → masked → black)
        masked_ood = np.ma.masked_where(gt == ignore_label, ood_map)
        cmap_masked = plt.cm.jet.copy()
        cmap_masked.set_bad(color="black")

        gt_only = gt.copy()

        # Mask wherever gt == 255:
        gt_masked = np.ma.masked_where(gt_only == ignore_label, gt_only)

        cmap_gt = ListedColormap(["darkgreen", "crimson"])  # 0→green, 1→red
        cmap_gt.set_bad(color="black")                       # 255→black

        plt.figure(figsize=(16, 12))

        # ---------------------------------------
        #  Panel (1,1): RGB + segmentation overlay
        # ---------------------------------------
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.imshow(seg_map, cmap="jet", alpha=0.4)
        if dots is not None:
            for (y, x, c) in dots:
                plt.scatter(x, y, s=10, c=c, marker="o",
                            edgecolors="white", linewidths=1, zorder=5)
        plt.axis("off")
        plt.title("RGB + Segmentation Overlay")

        # -------------------------------
        #  Panel (1,2): RAW OOD heatmap
        # -------------------------------
        plt.subplot(2, 2, 2)
        im_raw = plt.imshow(ood_map, cmap="jet", vmin=vmin, vmax=vmax)
        if dots is not None:
            for (y, x, c) in dots:
                plt.scatter(x, y, s=10, c=c, marker="o",
                            edgecolors="white", linewidths=1, zorder=5)
        plt.colorbar(im_raw, fraction=0.046, pad=0.04, format="%.2f")
        plt.axis("off")
        plt.title("Raw OOD Heatmap")

        # -----------------------------------------------
        #  Panel (2,1): OOD heatmap with ignore as black
        # -----------------------------------------------
        plt.subplot(2, 2, 3)
        im_masked = plt.imshow(masked_ood, cmap=cmap_masked, vmin=vmin, vmax=vmax)
        if dots is not None:
            for (y, x, c) in dots:
                plt.scatter(x, y, s=20, c=c, marker="o",
                            edgecolors="white", linewidths=1.5, zorder=5)
        plt.colorbar(im_masked, fraction=0.046, pad=0.04, format="%.2f")
        plt.axis("off")
        plt.title("OOD Heatmap (ignore=black)")

        # ------------------------------------
        #  Panel (2,2): Ground Truth only
        # ------------------------------------
        plt.subplot(2, 2, 4)
        im_gt = plt.imshow(gt_masked, cmap=cmap_gt, vmin=0, vmax=1)
        cbar2 = plt.colorbar(im_gt, ticks=[0, 1], fraction=0.046, pad=0.04)
        cbar2.set_label("GT label (0=ID, 1=OOD)")
        plt.axis("off")
        plt.title("Ground Truth (ignore=black)")

        plt.tight_layout()
        plt.show()

    def plot_pixel_activation(self, pixel: Tuple[int,int], kind: str,
                              img_shape: Tuple[int,int]):
        y, x = pixel; H_img, W_img = img_shape
        _, _, Hf, Wf = self.input.shape
        yf, xf = int(y*Hf/H_img), int(x*Wf/W_img)
        act = self.input[0, :, yf, xf].cpu().numpy()
        weights = (self.model.decode_head.conv_seg.weight
                   .data.squeeze(-1).squeeze(-1).cpu().numpy())
        pred_id = int(self.model_IDs_output[0, yf, xf])
        contrib = self.average_activations_per_class[pred_id].cpu().numpy()
        order   = np.argsort(contrib)
        plt.figure(figsize=(8, 4))
        plt.plot(act[order]*weights[pred_id][order], marker='o', ms=2)
        plt.title(f'{kind} pixel ({y},{x}) pred={pred_id}')
        plt.xlabel('channel rank'); plt.ylabel('activation'); plt.tight_layout(); plt.show()

    def plot_pair_ratio(self,
                        pixel: Tuple[int, int],
                        img_shape: Tuple[int, int],
                        cls:   int = None,
                        nbins: int = None):
        y_img, x_img       = pixel[:2]
        H_img, W_img       = img_shape
        _, _, Hf, Wf       = self.feature_map.shape
        yf = int(y_img * Hf / H_img)
        xf = int(x_img * Wf / W_img)

        # 2) extract feature-vector at that location
        v = self.feature_map[0, :, yf, xf].detach().cpu().numpy()   # [512]
        if cls is None:
            cls = int(self.model_IDs_output[0, yf, xf])

        # 3) raw ratio for every (i,j) pair
        raw     = np.outer(v, v).ravel()                            # [512²]
        mean_ij = self.mean_wt_any[cls].cpu().numpy().ravel()       # [512²]
        ratio   = raw / (mean_ij + 1e-9)

        # 4) order by rarity
        rarity = self.uncommon_any[cls].cpu().numpy().ravel()
        order  = np.argsort(rarity)[::-1]
        ratio_sorted = ratio[order]

        # 5) raw vs. binned plot
        if nbins is None or nbins <= 1:
            x = np.arange(ratio_sorted.size)
            y = ratio_sorted
            xlabel = "Channel-pair rank (rarest → most common)  [raw 262 144 points]"
        else:
            groups = np.array_split(ratio_sorted, nbins)
            y = np.array([g.mean() for g in groups])

            # map each bin back onto the original rank axis
            edges = np.linspace(0, ratio_sorted.size, nbins + 1)
            x     = 0.5 * (edges[1:] + edges[:-1])      # centre rank of each bin

            xlabel = "Channel-pair rank (rarest → most common)"

        # 6) draw
        plt.figure(figsize=(12, 4))
        plt.plot(x, y, lw=1)
        plt.axhline(1.0, color="k", ls="--", lw=1)
        plt.xlabel(xlabel)
        plt.ylabel(r"$(v_i v_j)\,/\,E[a_i a_j]$")
        plt.title(f"Pixel ({y_img},{x_img}), class {cls}")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------
    #  Metric computation
    # --------------------------------------------------------------
    def calculate_metrics(
            self,
            gt_ids: np.ndarray,
            scores: np.ndarray,
            device: str = "cuda",
            batch_size: int = 100_000,
            plot_hist: bool = False,              # ← NEW SWITCH
            nbins: int = 100,                     # ← bins for the histogram
            title: str = "OOD score distribution"
    ):
        """
        Calculates AUROC, FPR@95 % TPR and AP with TorchMetrics
        *and* (optionally) plots a histogram of the score distribution, as well
        as an average precision curve.
        """

        # ────────────────────────────────────────────────────────────────
        # 1. metrics 
        g_all = torch.from_numpy(gt_ids).to(device)
        s_all = torch.from_numpy(scores).float().to(device)

        s_all = s_all - s_all.min()
        s_all = s_all / s_all.max()

        auroc = BinaryAUROC().to(device)
        ap    = BinaryAveragePrecision().to(device)
        roc   = BinaryROC().to(device)

        for i in range(0, g_all.numel(), batch_size):
            auroc.update(s_all[i:i+batch_size], g_all[i:i+batch_size].int())
            ap.update(   s_all[i:i+batch_size], g_all[i:i+batch_size].int())
            roc.update(  s_all[i:i+batch_size], g_all[i:i+batch_size].int())

        auc  = auroc.compute().item()
        ap_v = ap.compute().item()
        fpr, tpr, _ = roc.compute()
        fpr95 = fpr[tpr >= 0.95][0].item()

        print(f"AUROC={auc:.4f}  FPR95={fpr95:.4f}  AP={ap_v:.4f}")

        # ────────────────────────────────────────────────────────────────
        # histogram
        if plot_hist:
            id_scores  = scores[gt_ids == 0]
            ood_scores = scores[gt_ids == 1]

            plt.figure(figsize=(6, 4))
            plt.hist(id_scores,  bins=nbins, alpha=0.6, density=True,
                    label=f'ID  (n={len(id_scores):,})')
            plt.hist(ood_scores, bins=nbins, alpha=0.6, density=True,
                    label=f'OOD (n={len(ood_scores):,})')
            plt.xlabel("OOD score");  plt.ylabel("Density")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_pr = False
        if plot_pr:

            # gt_ids is a NumPy array of 0 (ID) / 1 (OOD), scores is the OOD‐score
            precision, recall, _ = precision_recall_curve(gt_ids, scores)
            ap_scalar = average_precision_score(gt_ids, scores)

            plt.figure(figsize=(6, 4))
            plt.plot(
                recall,
                precision,
                lw=1.5,
                label=f"PR curve (AP={100*ap_scalar:.2f}%)"
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall Curve")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.show()

        return auc, fpr95, ap_v

    def _calculate_coactivation_score(
        self,
        labels: np.ndarray,
        common_inverse: torch.Tensor,
        uncommon_inverse: torch.Tensor,
        chunk_size: int = 4096,
        abs_logits: bool = False
    ):
        """
        Returns an OOD-heat-map where each class’s co-activation ratio
        is weighted by either its softmax probability
        or absolute logit, then summed over classes.
        """
        if self.feature_map is None:
            raise RuntimeError("feature_map not cached by hook!")

        # dimensions & device
        _, _, Hf, Wf = self.feature_map.shape
        H_img, W_img = labels.shape
        N            = Hf * Wf
        dev          = self.device
        eps          = 1e-9

        C = self.feature_map.size(1)           # e.g. 512
        K = self.model_output.size(1)          # number of classes, e.g. 19

        # Flatten feature map → [N,C], logits → [N,K], all on GPU
        feat = (
            self.feature_map
                .squeeze(0)                    # [C,Hf,Wf]
                .permute(1,2,0)                # [Hf,Wf,C]
                .reshape(-1, C)                # [N,C]
                .to(dev, non_blocking=True)
        )
        if self.make_feat_binary == True:
            feat = (feat > 0).float()

        logits_flat = (
            self.model_output
                .squeeze(0)                    # [K,Hf,Wf]
                .permute(1,2,0)                # [Hf,Wf,K]
                .reshape(-1, K)                # [N,K]
                .to(dev, non_blocking=True)
        )

        # Choose weights: softmax probs or abs(logit)
        if self.weight_mode == 'softmax':
            w_all = torch.softmax(logits_flat / self.temperature_co, dim=1)

        elif self.weight_mode == 'logits':
            # raw logits
            w_all = logits_flat

        elif self.weight_mode == 'pred':
            # Only predicted class
            pred_ids = torch.argmax(logits_flat, dim=1)                # [N]
            w_all = torch.zeros_like(logits_flat)
            w_all.scatter_(1, pred_ids.unsqueeze(1), 1.0)              # one-hot per row
        else:
            raise ValueError(f"unknown weight_mode={self.weight_mode}")

        # 3) Prepare binary masks & normalisers on GPU
        if self.inverse_convert_to_ones == True:
            M_c = (common_inverse > 0).to(dev).float()       # [K,C,C]
            M_u = (uncommon_inverse > 0).to(dev).float()     # [K,C,C]
        else:
            M_c = common_inverse.to(dev).float()       # [K,C,C]
            M_u = uncommon_inverse.to(dev).float()     # [K,C,C]
        # n_c = M_c.view(K, -1).sum(1).clamp_min(eps)   # [K] # Class normalisation that is WRONG, but can actually improve performance(cheating)
        # n_u = M_u.view(K, -1).sum(1).clamp_min(eps)   # [K] # Class normalisation that is WRONG, but can actually improve performance(cheating)

        # flatten masks for GEMM
        M_c2 = M_c.view(K*C, C)  # [K*C, C]
        M_u2 = M_u.view(K*C, C)

        ood_out = torch.empty(N, device=dev)

        # Chunked GPU loop
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            v = feat[s:e]                           # [B, C]
            B = v.size(0)

            # project v onto each class-inverse-matrix
            pc = (v @ M_c2.T).view(B, K, C)         # [B, K, C]
            pu = (v @ M_u2.T).view(B, K, C)         # [B, K, C]

            # co-activation sums
            s_c = (v.unsqueeze(1) * pc).sum(dim=2)# / n_c   # [B, K]
            s_u = (v.unsqueeze(1) * pu).sum(dim=2)# / n_u   # [B, K]

            if self.c_u_ratio == True:
                co_ratio = s_u/s_c # [B, K]
            else:
                co_ratio = s_u     # [B, K]

            # weight, sum over classes → [B], clamp
            score = (co_ratio * w_all[s:e]).sum(dim=1)                       # [B]
            if self.clips != (0, 0):
                score = torch.clamp(score, min=self.clips[0], max=self.clips[1]) # [B]
            ood_out[s:e] = score

        ood_map_gpu = ood_out.view(1, 1, Hf, Wf)          # [1,1,Hf,Wf]

        if self.max_pool != 0:
            ood_map_gpu = F.max_pool2d(ood_map_gpu, kernel_size=self.max_pool, stride=1, padding=self.max_pool//2)
        if self.co_blur_ksize != 0:
            k = self.co_blur_ksize
            σ = float(self.co_sigma)
            if σ <= 0:
                σ = 1

            coords = torch.arange(k, device=dev) - (k // 2)
            g1d = torch.exp(-coords.float().pow(2) / (2 * σ * σ))
            g1d /= g1d.sum()
            kernel2d = (g1d[:, None] @ g1d[None, :]).view(1, 1, k, k).to(dev)
            ood_map_gpu = F.conv2d(ood_map_gpu, kernel2d, padding=k // 2)

        ood_map = ood_map_gpu.squeeze().cpu().numpy()    # [Hf,Wf]
        # np.log(ood_map, out=ood_map) #if they have to big differences

        # Uncomment if you want to do divide by another map and do additional blurring after
        # ood_map = -ood_map/self.use_elsewhere_map/self.use_elsewhere_map/self.use_elsewhere_map
        # do numpy gaussian blur here

        ood_map = cv2.resize(
            ood_map,
            (W_img, H_img),                             # width, height
            interpolation=cv2.INTER_NEAREST
        )
        return ood_map
