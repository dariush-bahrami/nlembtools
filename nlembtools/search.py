from typing import Iterable, Literal, NamedTuple, Optional

import torch

from .core import FeatureExtractor


class MatchedResult(NamedTuple):
    index: int
    score: float


class TextSearchResult(NamedTuple):
    text: str
    score: float


def cosine_semantic_match(
    query_features: torch.Tensor,
    key_features: torch.Tensor,
    top_k: int = 1,
) -> list[list[MatchedResult]]:
    """Semantic matching between query and key features. The matching is done by
    computing the cosine similarity between the query and key features.

    Args:
        query_features (torch.Tensor): A tensor of shape (num_queries, feature_dim)
            containing the query features.
        key_features (torch.Tensor): A tensor of shape (num_keys, feature_dim)
            containing the key features.
        top_k (int, optional): The number of top matches to return. Defaults to 1.

    Returns:
        list[list[MatchedResult]]: A list of lists of MatchedResult objects. The
            outer list has length num_queries and the inner list has length top_k.
    """
    q_norms = query_features.norm(dim=1, keepdim=True, p=2)
    k_norms = key_features.norm(dim=1, keepdim=True, p=2)
    norm_prods = q_norms @ k_norms.T
    cosine_similarities = (query_features @ key_features.T) / norm_prods
    matched_keys_indices = cosine_similarities.argsort(dim=1, descending=True).tolist()
    resutls = []
    for i, j in enumerate(matched_keys_indices):
        r = []
        for k in range(top_k):
            r.append(MatchedResult(j[k], cosine_similarities[i, j[k]].item()))
        resutls.append(r)
    return resutls


def euclidean_semantic_match(
    query_features: torch.Tensor,
    key_features: torch.Tensor,
    top_k: int = 1,
) -> list[list[MatchedResult]]:
    """Semantic matching between query and key features. The matching is done by
    computing the euclidean distance between the query and key features.

    Args:
        query_features (torch.Tensor): A tensor of shape (num_queries, feature_dim)
            containing the query features.
        key_features (torch.Tensor): A tensor of shape (num_keys, feature_dim)
            containing the key features.
        top_k (int, optional): The number of top matches to return. Defaults to 1.

    Returns:
        list[list[MatchedResult]]: A list of lists of MatchedResult objects. The
            outer list has length num_queries and the inner list has length top_k.
    """
    distances = torch.cdist(query_features, key_features, p=2)
    similarities = 1 / (1 + distances)
    matched_keys_indices = distances.argsort(dim=1).tolist()
    resutls = []
    for i, j in enumerate(matched_keys_indices):
        r = []
        for k in range(top_k):
            r.append(MatchedResult(j[k], similarities[i, j[k]].item()))
        resutls.append(r)
    return resutls


class SemanticTextSearch:
    def __init__(
        self,
        model_name_or_path: str,
        strings: Iterable[str],
        matching_method: Literal["cosine", "euclidean"] = "cosine",
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.feature_extractor = FeatureExtractor(model_name_or_path, device)
        self.device = self.feature_extractor.device
        self.strings = strings
        self.features = self.feature_extractor(strings)
        self.matching_method = matching_method

        if self.matching_method == "cosine":
            self.matching_function = cosine_semantic_match
        elif self.matching_method == "euclidean":
            self.matching_function = euclidean_semantic_match
        else:
            raise Exception(f"Invalid matching method: {self.matching_method}")

    def append_strings(self, strings: list[str]):
        self.strings.extend(strings)
        self.features = torch.cat(
            [self.features, self.feature_extractor(strings)],
            dim=0,
        )

    def remove_strings(self, strings: list[str]):
        # find indices of strings to remove
        indices_to_remove = []
        for s in strings:
            indices_to_remove.append(self.strings.index(s))
        # remove strings and features
        self.strings = [
            s for i, s in enumerate(self.strings) if i not in indices_to_remove
        ]
        indices_to_keep = [
            i for i in range(len(self.strings)) if i not in indices_to_remove
        ]
        self.features = torch.cat(
            [self.features[i].unsqueeze(0) for i in indices_to_keep],
            dim=0,
        )

    def __call__(self, queries: list[str], top_k: int = 1) -> list[str]:
        q_features = self.feature_extractor(queries)
        k_features = self.features
        matching_results = self.matching_function(q_features, k_features, top_k)
        resutls = []
        for matched_result_list in matching_results:
            query_text_searchs_results = []
            for matched_result in matched_result_list:
                query_text_searchs_results.append(
                    TextSearchResult(
                        self.strings[matched_result.index],
                        matched_result.score,
                    )
                )
            resutls.append(query_text_searchs_results)
        return resutls

    def __repr__(self):
        return f"SemanticTextSearch({self.model_name_or_path})"
