"""Key-frame extraction utilities for OD matrix sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim


class KeyFrameExtractor(object):
    """Select representative frames from an OD matrix sequence."""

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_key_frame_indices_to_csv(self, key_frame_indices, filename="key_frame_indices.csv"):
        path = self.output_dir / filename
        df = pd.DataFrame(key_frame_indices, columns=["key_frame_index"])
        df.to_csv(path, index=False)
        print(f"Key-frame indices saved to {path}")

    def save_mssim_series_to_csv(
        self,
        ssim_arr,
        threshold,
        quantile,
        filename="mssim_series.csv",
    ):
        path = self.output_dir / filename
        t_idx = np.arange(len(ssim_arr))
        is_discontinuity = (ssim_arr < threshold).astype(int)

        df = pd.DataFrame(
            {
                "t": t_idx,
                "mssim": ssim_arr,
                "threshold": np.full_like(ssim_arr, threshold, dtype=float),
                "is_discontinuity": is_discontinuity,
                "selected_keyframe": is_discontinuity,
                "quantile": np.full_like(ssim_arr, quantile, dtype=float),
            }
        )
        df.to_csv(path, index=False)
        print(f"MSSIM series saved to {path}")

    def linear_discontinuity_search(self, data, quantile=0.4):
        """Detect key frames with quantile-thresholded pairwise SSIM."""
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1].")

        data = self._ensure_channel_dim(data)
        num_frames = data.shape[0]
        if num_frames == 0:
            raise ValueError("Cannot extract key frames from an empty sequence.")
        if num_frames == 1:
            key_frame_indices = [0]
            self.save_key_frame_indices_to_csv(key_frame_indices)
            return key_frame_indices

        ssim_arr = self._pairwise_ssim(data)
        adaptive_threshold = np.quantile(ssim_arr, quantile)
        self.save_mssim_series_to_csv(
            ssim_arr=ssim_arr,
            threshold=adaptive_threshold,
            quantile=quantile,
            filename=f"mssim_series_q{quantile:.2f}.csv",
        )

        key_frame_indices = [0]
        for t, similarity in enumerate(ssim_arr):
            if similarity < adaptive_threshold:
                key_frame_indices.append(t + 1)

        if key_frame_indices[-1] != num_frames - 1:
            key_frame_indices.append(num_frames - 1)

        key_frame_indices = sorted(set(key_frame_indices))
        self.save_key_frame_indices_to_csv(key_frame_indices)
        print(
            f"[KF-Quantile] q={quantile:.2f}, "
            f"threshold={adaptive_threshold:.6f}, "
            f"actual_ratio={len(key_frame_indices) / num_frames:.3f}"
        )
        return key_frame_indices

    def linear_discontinuity_search_cpd(self, data, pen=5.0, ensure_last=True):
        """Detect key frames with PELT change-point detection over SSIM distance."""
        try:
            import ruptures as rpt
        except ImportError as exc:
            raise ImportError(
                "ruptures is required for CPD key-frame extraction. "
                "Install it or use linear_discontinuity_search instead."
            ) from exc

        data = self._ensure_channel_dim(data)
        num_frames = data.shape[0]
        if num_frames == 0:
            raise ValueError("Cannot extract key frames from an empty sequence.")
        if num_frames == 1:
            return [0]

        ssim_arr = self._pairwise_ssim(data)
        dissimilarity = (1.0 - ssim_arr).reshape(-1, 1)

        algo = rpt.Pelt(model="l2").fit(dissimilarity)
        breakpoints = algo.predict(pen=pen)

        key_frame_indices = [0]
        for breakpoint in breakpoints:
            if breakpoint < (num_frames - 1):
                key_frame_indices.append(breakpoint + 1)

        if ensure_last:
            key_frame_indices.append(num_frames - 1)

        key_frame_indices = sorted(set(key_frame_indices))
        self.save_key_frame_indices_to_csv(key_frame_indices)
        print(
            f"[KF-CPD] pen={pen}, num_kf={len(key_frame_indices)}, "
            f"ratio={len(key_frame_indices) / num_frames:.3f}"
        )
        return key_frame_indices

    @staticmethod
    def _ensure_channel_dim(data):
        data = np.asarray(data)
        if data.ndim == 3:
            return data[:, :, :, np.newaxis]
        if data.ndim == 4:
            return data
        raise ValueError("OD data must have shape (T, N, N) or (T, N, N, C).")

    @staticmethod
    def _pairwise_ssim(data):
        ssim_list = []
        for t in range(data.shape[0] - 1):
            frame1 = data[t, :, :, 0]
            frame2 = data[t + 1, :, :, 0]

            data_range = frame2.max() - frame2.min()
            if data_range < 1e-8:
                data_range = 1.0

            ssim_list.append(ssim(frame1, frame2, data_range=data_range))

        return np.asarray(ssim_list, dtype=np.float64)
