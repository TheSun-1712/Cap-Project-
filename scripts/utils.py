"""High-performance utility optimizations for Anomalib on modern GPUs.

Enables GPU-accelerated Coreset Subsampling for PatchCore on Blackwell (RTX 5060).
Anomalib's default KCenterGreedy runs distance updates on CPU, causing coreset
selection to take 10+ minutes on large industrial datasets with 1,200+ images.
This GPU patch moves the transformed features to CUDA, completing the selection
in 2-5 seconds.
"""

import torch
from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy


def patch_gpu_coreset():
    """Monkey-patch KCenterGreedy to run vectorized distance updates directly on CUDA."""
    orig_select = KCenterGreedy.select_coreset_idxs

    def fast_select_coreset_idxs(self) -> list[int]:
        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.reset_distances()

        if torch.cuda.is_available():
            self.features = self.features.cuda()

        idx = torch.randint(high=self.n_observations, size=(1,), device=self.features.device).squeeze()
        selected_coreset_idxs: list[torch.Tensor] = [idx]

        for _ in range(self.coreset_size - 1):
            self.update_distances(cluster_center=idx)
            idx = self.get_new_idx()
            self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), 0.0)
            selected_coreset_idxs.append(idx)

        return torch.stack(selected_coreset_idxs).cpu().tolist()

    KCenterGreedy.select_coreset_idxs = fast_select_coreset_idxs


# Automatically apply on import
patch_gpu_coreset()
