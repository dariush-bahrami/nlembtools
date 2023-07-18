from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer


class FeatureExtractor:
    """A wrapper class for feature extraction from a pretrained model.

    Args:
        model_name_or_path (str): The name or path of the pretrained model. This will
            be passed to `AutoModel.from_pretrained` and `AutoTokenizer.from_pretrained`
            from the `transformers` library.
        device (str, optional): The device to use for the model. Defaults to "cuda" if
            available, otherwise "cpu".
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Using device:", device)

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, strings: list[str]) -> torch.Tensor:
        """Extract features from a list of strings.

        Args:
            strings (list[str]): A list of strings to extract features from.

        Returns:
            torch.Tensor: A tensor of shape (len(strings), feature_dim) containing the
                extracted features.
        """
        # tokenization
        inputs = self.tokenizer(
            strings,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # hidden states extraction
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_output = self.model(**inputs)
        hidden_states = model_output.last_hidden_state
        # mean pooling
        features = hidden_states.mean(dim=1)
        return features

    def __repr__(self):
        return f"BERTFeatureExtractor({self.model_name_or_path})"
