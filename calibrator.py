# calibration.py – Calibrator for LINe and pair-wise co-activation stats
# =====================================================================
# Author: Victor Lindholm
# This module gathers statistics that are later used for OOD detection.

from __future__ import annotations

from typing import List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.apis import inference_segmentor

# ----------------------------------------------------------------------
#  A fixed raw-to-train LUT for the Cityscapes labels
# ----------------------------------------------------------------------
_RAW2TRAIN: dict[int, int] = {
    7: 0,  8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_LUT = np.full(256, 255, np.uint8)          # 255 = “ignore”
for raw_id, train_id in _RAW2TRAIN.items():
    _LUT[raw_id] = train_id


# ======================================================================
#  Calibrator
# ======================================================================
class Calibrator:
    """Collect LINe tensors *and* pair-wise co-activation statistics."""

    def __init__(
        self,
        mmseg_model: nn.Module,            # ← pass a *ready* MMSeg model
        device: str | None = None,
        n_classes: int = 19,
        n_channels: int = 512,
    ) -> None:
        self.model = mmseg_model
        self.dev   = device or next(mmseg_model.parameters()).device
        self.K, self.C = n_classes, n_channels

        # ---------- LINe tensors ----------
        self.contrib = torch.zeros(self.K, self.C, device=self.dev)
        self.avg_act = torch.zeros(self.K, self.C, device=self.dev)

        # ---------- per-class accumulators ----------
        sz = (self.K, self.C, self.C)
        self.co_bin_count   = torch.zeros(sz, device=self.dev)
        self.co_wt_sum      = torch.zeros(sz, device=self.dev)
        self.co_wt_sum_both = torch.zeros(sz, device=self.dev)
        self.pixel_counter  = torch.zeros(self.K, device=self.dev)

        # ---------- global accumulators ----------
        self.co_bin_all     = torch.zeros(self.C, self.C, device=self.dev)
        self.co_wt_all      = torch.zeros(self.C, self.C, device=self.dev)
        self.co_wt_both_all = torch.zeros(self.C, self.C, device=self.dev)
        self.pixel_all      = torch.tensor(0.0, device=self.dev)

    # ------------------------------------------------------------------
    #  Main dataset pass
    # ------------------------------------------------------------------
    def run(self, images: List[str], gt_paths: List[str]) -> None:
        """Iterate over *images* and *gt_paths* to fill all statistics."""

        hook_buf: dict[str, torch.Tensor] = {}

        def _cache_feat(_, inp, __):
            hook_buf["feat"] = inp[0].detach()          # [1,C,Hf,Wf]

        self.model.decode_head.conv_seg.register_forward_hook(_cache_feat)
        w = self.model.decode_head.conv_seg.weight.data.squeeze(-1).squeeze(-1)  # [K,C]

        for idx, (img_p, gt_p) in enumerate(zip(images, gt_paths), 1):
            if idx % 25 == 0:
                print(f"[Calibrator] processed {idx}/{len(images)} images …")

            _ = inference_segmentor(self.model, img_p)           # runs model
            feat = hook_buf.pop("feat").squeeze(0)               # [C,Hf,Wf]
            Hf, Wf = feat.shape[1:]

            # ------------------ prepare ground truth ---------------------
            gt_raw   = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
            gt_train = _LUT[gt_raw]                              # 0-18 / 255
            valid    = (gt_train != 255).astype(np.uint8)

            gt_t = torch.from_numpy(gt_train)[None, None].float().to(self.dev)
            gt_t = F.interpolate(gt_t, size=(Hf, Wf), mode="nearest").long().squeeze()

            val_t = torch.from_numpy(valid)[None, None].float().to(self.dev)
            val_t = F.interpolate(val_t, size=(Hf, Wf), mode="nearest").bool().squeeze()

            # ------------------ accumulate statistics -------------------
            for cls in range(self.K):
                mask = (gt_t == cls) & val_t
                if not mask.any():
                    continue

                vecs = feat.permute(1, 2, 0)[mask]      # [N,C]
                binm = (vecs > 0).float()               # [N,C]
                n    = vecs.size(0)

                # per-class
                self.co_bin_count[cls]   += binm.T @ binm
                self.co_wt_sum[cls]      += vecs.T @ vecs
                self.co_wt_sum_both[cls] += (vecs * binm).T @ (vecs * binm)
                self.pixel_counter[cls]  += n

                # global
                self.co_bin_all     += binm.T @ binm
                self.co_wt_all      += vecs.T @ vecs
                self.co_wt_both_all += (vecs * binm).T @ (vecs * binm)
                self.pixel_all      += n

                # LINe tensors
                self.contrib[cls] += (vecs.sum(0) * w[cls]).abs()
                self.avg_act[cls] += (vecs > 0).sum(0)

        # ------------------ normalisation -------------------------------
        eps = 1e-9
        pix_cls = self.pixel_counter.clamp_min(eps).view(self.K, 1)
        self.contrib /= pix_cls
        self.avg_act /= pix_cls

        pc = self.pixel_counter.view(self.K, 1, 1) + eps
        self.freq_bin     = self.co_bin_count   / pc
        self.mean_wt      = self.co_wt_sum      / pc
        self.mean_wt_both = self.co_wt_sum_both / (self.co_bin_count + eps)

        diag = torch.diagonal(self.co_bin_count, dim1=1, dim2=2)
        cnt_any = diag.unsqueeze(2) + diag.unsqueeze(1) - self.co_bin_count
        self.mean_wt_any = self.co_wt_sum / (cnt_any + eps)

        self.freq_bin_all     = self.co_bin_all     / (self.pixel_all + eps)
        self.mean_wt_all      = self.co_wt_all      / (self.pixel_all + eps)
        self.mean_wt_both_all = self.co_wt_both_all / (self.co_bin_all + eps)

    # ------------------------------------------------------------------
    #  (De)serialisation helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return everything needed to recreate the calibrator later."""
        return {
            "model_weights":       self.model.state_dict(),
            "contrib":             self.contrib.cpu(),
            "avg_act":             self.avg_act.cpu(),
            "freq_bin":            self.freq_bin.cpu(),
            "mean_wt":             self.mean_wt.cpu(),
            "mean_wt_both":        self.mean_wt_both.cpu(),
            "mean_wt_any":         self.mean_wt_any.cpu(),
            "freq_bin_all":        self.freq_bin_all.cpu(),
            "mean_wt_all":         self.mean_wt_all.cpu(),
            "mean_wt_both_all":    self.mean_wt_both_all.cpu(),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"[Calibrator] saved → {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str,
        mmseg_cfg: str,
        mmseg_ckpt: str,
        device: str | None = None,
    ) -> "Calibrator":
        """Recreate a Calibrator from disk plus a fresh MMSeg model."""
        from mmseg.apis import init_segmentor

        raw_model = init_segmentor(mmseg_cfg, mmseg_ckpt,
                                   device=device or 'cpu')
        obj  = cls(raw_model, device)
        data = torch.load(path, map_location=obj.dev)

        for k, v in data.items():
            if k == "model_weights":
                continue
            setattr(obj, k, v.to(obj.dev))

        raw_model.load_state_dict(data["model_weights"], strict=False)
        print(f"[Calibrator] loaded ← {path}")
        return obj
