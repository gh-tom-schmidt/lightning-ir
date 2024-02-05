import pytest
from transformers import AutoModel

from tide.datamodule import DataModule
from tide.mvr import MVRConfig, MVRModel, MVRModule
from tide.loss import LocalizedContrastive, MarginMSE, RankNet


@pytest.fixture(scope="module")
def mvr_model(model_name_or_path: str) -> MVRModel:
    model = AutoModel.from_pretrained(model_name_or_path)
    model.config.update(MVRConfig().to_diff_dict())
    model.resize_token_embeddings(model.config.vocab_size, 8)
    return MVRModel(model)


@pytest.fixture(scope="module")
def margin_mse_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, MarginMSE())


@pytest.fixture(scope="module")
def ranknet_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, RankNet())


@pytest.fixture(scope="module")
def localized_contrastive_module(mvr_model: MVRModel) -> MVRModule:
    return MVRModule(mvr_model, LocalizedContrastive())


def test_doc_padding(relevance_run_datamodule: DataModule, mvr_model: MVRModel):
    batch = next(iter(relevance_run_datamodule.train_dataloader()))
    model = mvr_model
    doc_encoding = batch.doc_encoding
    doc_encoding["input_ids"] = doc_encoding["input_ids"][:-1]
    doc_encoding["attention_mask"] = doc_encoding["attention_mask"][:-1]
    doc_encoding["token_type_ids"] = doc_encoding["token_type_ids"][:-1]

    query_embedding = model.encode(**batch.query_encoding)
    doc_embedding = model.encode(**batch.doc_encoding)
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            batch.query_encoding.attention_mask,
            doc_embedding,
            batch.doc_encoding.attention_mask,
            None,
        )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            batch.query_encoding.attention_mask,
            doc_embedding,
            batch.doc_encoding.attention_mask,
            [doc_embedding.shape[0]],
        )
    with pytest.raises(ValueError):
        model.score(
            query_embedding,
            batch.query_encoding.attention_mask,
            doc_embedding,
            batch.doc_encoding.attention_mask,
            [0] * query_embedding.shape[0],
        )

    num_docs = [len(docs) for docs in batch.doc_ids]
    num_docs[-1] = num_docs[-1] - 1
    scores = model.score(
        query_embedding,
        batch.query_encoding.attention_mask,
        doc_embedding,
        batch.doc_encoding.attention_mask,
        num_docs,
    )
    assert (scores == model.scoring_function.MASK_VALUE).sum() == 1


def test_margin_mse(margin_mse_module: MVRModule, triples_datamodule: DataModule):
    dataloader = triples_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = margin_mse_module.training_step(batch, 0)
    assert loss


def test_ranknet(ranknet_module: MVRModule, rank_run_datamodule: DataModule):
    dataloader = rank_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = ranknet_module.training_step(batch, 0)
    assert loss


def test_localized_contrastive(
    localized_contrastive_module: MVRModule,
    single_relevant_run_datamodule: DataModule,
):
    dataloader = single_relevant_run_datamodule.train_dataloader()
    batch = next(iter(dataloader))
    loss = localized_contrastive_module.training_step(batch, 0)
    assert loss


def test_validation_step(
    margin_mse_module: MVRModule,
    relevance_run_datamodule: DataModule,
):
    dataloader = relevance_run_datamodule.val_dataloader()[0]
    batch = next(iter(dataloader))
    margin_mse_module.validation_step(batch, 0, 0)
    outputs = margin_mse_module.validation_step_outputs
    assert len(outputs) == 2
    assert outputs[0][0] == "ndcg@10"
    assert outputs[1][0] == "mrr@ranking"
