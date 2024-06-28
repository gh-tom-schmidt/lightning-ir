from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import MODEL_MAPPING, AutoConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import load_state_dict

from ...base import LightningIRModel, LightningIRModelClassFactory
from ...bi_encoder import BiEncoderModel
from .config import SpladeConfig


class MLMHead(torch.nn.Module):
    def __init__(self, config: SpladeConfig) -> None:
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SpladeModel(BiEncoderModel):
    config_class = SpladeConfig

    def __init__(self, config: SpladeConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: SpladeConfig

        if self.config.projection == "mlm":
            self.projection = MLMHead(config)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str | Path, *args, **kwargs
    ) -> LightningIRModel:
        if "naver" in str(model_name_or_path):
            return cls.from_naver_checkpoint(model_name_or_path)
        return super().from_pretrained(model_name_or_path, *args, **kwargs)

    @classmethod
    def from_naver_checkpoint(
        cls, model_name_or_path: str | Path, *args, **kwargs
    ) -> LightningIRModel:

        config = AutoConfig.from_pretrained(model_name_or_path)
        BackboneModel = MODEL_MAPPING[config.__class__]
        cls = LightningIRModelClassFactory(BackboneModel, SpladeConfig)
        config = cls.config_class.from_pretrained(model_name_or_path)
        config.update({})
        model = cls(config=config, add_pooling_layer=False)
        state_dict_path = hf_hub_download(
            repo_id=str(model_name_or_path), filename="pytorch_model.bin"
        )
        state_dict = load_state_dict(state_dict_path)
        state_dict.pop("bert.embeddings.position_ids", None)
        state_dict.pop("cls.predictions.bias", None)
        for key in list(state_dict.keys()):
            if key.startswith("bert."):
                state_dict[key[5:]] = state_dict.pop(key)
            else:
                new_key = key.replace("cls.predictions", "projection").replace(
                    ".transform", ""
                )
                state_dict[new_key] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        return model