import torch
from transformers import BatchEncoding
from transformers.modeling_utils import cached_file, load_state_dict

from ...base.model import LightningIRModel
from ...cross_encoder.model import CrossEncoderModel, CrossEncoderOutput
from .config import T5CrossEncoderConfig


class ScaleLinear(torch.nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        input = input * (input.shape[-1] ** -0.5)
        return super().forward(input)


class T5CrossEncoderModel(CrossEncoderModel):
    config_class = T5CrossEncoderConfig

    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "linear.weight"]

    def __init__(self, config: T5CrossEncoderConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: T5CrossEncoderConfig
        if self.config.decoder_strategy == "mono":
            self.linear = ScaleLinear(config.hidden_size, 2)
        else:
            self.linear = ScaleLinear(config.hidden_size, 1, bias=config.linear_bias)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs) -> LightningIRModel:
        instance = super().from_pretrained(model_name_or_path, *args, **kwargs)
        if instance.config.decoder_strategy == "mono":
            # [1176, 6136] true, false
            weights = instance.shared.weight.data[[1176, 6136]]
        elif instance.config.decoder_strategy == "rank":
            # 32089 <extra_id_10>
            weights = instance.shared.weight.data[[32089]]
        else:
            raise ValueError("Unknown decoder strategy")
        instance.linear.weight.data = weights
        return instance

    # def get_output_embeddings(self):
    #     # TODO tieing of weights not working when setting linear to only use slice of lm head
    #     shared = self.shared
    #     if self.config.decoder_strategy == "mono":
    #         self.linear.weight.data = shared.weight.data[[1176, 6136]]
    #     elif self.config.decoder_strategy == "rank":
    #         self.linear.weight.data = shared.weight.data[[32089]]
    #     else:
    #         raise ValueError("Unknown decoder strategy")
    #     return shared

    def forward(self, encoding: BatchEncoding) -> CrossEncoderOutput:
        decoder_input_ids = torch.zeros(
            (encoding["input_ids"].shape[0], 1), device=encoding["input_ids"].device, dtype=torch.long
        )
        encoding["decoder_input_ids"] = decoder_input_ids
        output = super().forward(encoding)
        if output.scores is None:
            raise ValueError("Scores are None")
        if self.config.decoder_strategy == "mono":
            scores = output.scores.view(-1, 2)
            scores = torch.nn.functional.log_softmax(scores, dim=-1)[:, 0]
            output.scores = scores.view(-1)
        return output