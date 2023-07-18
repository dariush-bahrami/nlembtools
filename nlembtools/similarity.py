from typing import Optional

import torch

from .core import FeatureExtractor
from .datatypes import Triplet


class SimilarityMatrixExtractor:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.feature_extractor = FeatureExtractor(model_name_or_path, device)

    @torch.no_grad()
    def __call__(self, sentences: list[str]) -> torch.Tensor:
        features = self.feature_extractor(sentences)
        norms = features.norm(dim=1, keepdim=True, p=2)
        norm_prods = norms @ norms.T
        cosine_similarities = (features @ features.T) / norm_prods
        return cosine_similarities.cpu()


def extract_triplets(
    strings,
    similarity_matrix,
    negative_quantile,
    positive_quantile,
) -> set[Triplet]:
    triplets = set()
    for i in range(len(strings)):
        similarity_vector = similarity_matrix[i]
        q = torch.tensor([negative_quantile, positive_quantile])
        negative_threshold, positive_threshold = similarity_vector.quantile(q)
        indices = torch.arange(len(strings))
        index_mask = (similarity_vector <= negative_threshold) & (indices != i)
        negative_indices = indices[index_mask]
        index_mask = (similarity_vector >= positive_threshold) & (indices != i)
        positive_indices = indices[index_mask]
        for j in positive_indices:
            for k in negative_indices:
                triplets.add(Triplet(strings[i], strings[j], strings[k]))
    return triplets
