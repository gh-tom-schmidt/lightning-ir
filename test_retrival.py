import torch

from lightning_ir import BiEncoderModule

from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import ColBERTConfig, colbert_score

# basic test
query = "What is the capital of France?"
documents = [
    "Paris is the capital of France.",
    "France is a country in Europe.",
    "The Eiffel Tower is in Paris.",
]

model_name = "colbert-ir/colbertv2.0"
module = BiEncoderModule(model_name).eval()
with torch.inference_mode():
    output = module.score(query, documents)

# get best document
print(documents[torch.argmax(output.scores).item()])

# ColBERT
colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(model_name))
orig_model = Checkpoint(model_name, colbert_config)

orig_query = orig_model.queryFromText([query])
orig_docs = orig_model.docFromText(documents)

d_mask = ~(orig_docs == 0).all(-1)

orig_scores = colbert_score(orig_query, orig_docs, d_mask)

# get best document
print(documents[torch.argmax(orig_scores).item()])