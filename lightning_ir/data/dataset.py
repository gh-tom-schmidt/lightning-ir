import warnings
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, Literal, NamedTuple, Tuple

import ir_datasets
import ir_datasets.docs
import pandas as pd
import torch
from ir_datasets.formats import GenericDoc, GenericDocPair
from torch.distributed import get_rank, get_world_size
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .data import DocSample, QuerySample, RunSample, ScoredDocTuple

DASHED_DATASET_MAP = {
    dataset.replace("/", "-"): dataset for dataset in ir_datasets.registry._registered
}
RUN_HEADER = ["query_id", "q0", "doc_id", "rank", "score", "system"]


class QueryDatasetConfig(NamedTuple):
    num_queries: int | None = None


class DocDatasetConfig(NamedTuple):
    num_docs: int | None = None


class RunDatasetConfig(NamedTuple):
    targets: Literal["relevance", "subtopic_relevance", "rank", "score"] | None
    depth: int
    sample_size: int
    sampling_strategy: Literal["single_relevant", "top"]


class TupleDatasetConfig(NamedTuple):
    targets: Literal["order", "score"] | None
    num_docs: int | None


class DataParallelIterableDataset(IterableDataset):
    # https://github.com/Lightning-AI/pytorch-lightning/issues/15734
    def __init__(
        self, dataset: str, config: QueryDatasetConfig | DocDatasetConfig
    ) -> None:
        super().__init__()
        # TODO add support for multi-gpu and multi-worker inference; currently
        # doesn't work
        self.ir_dataset: ir_datasets.Dataset = ir_datasets.load(dataset)
        self.config = config
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        try:
            world_size = get_world_size()
            process_rank = get_rank()
        except (RuntimeError, ValueError):
            world_size = 1
            process_rank = 0

        self.num_replicas = num_workers * world_size
        self.rank = process_rank * num_workers + worker_id
        if isinstance(config, QueryDatasetConfig):
            self._field = "queries"
            self._iterator = self.ir_dataset.queries_iter
            self._sample_cls = QuerySample
        elif isinstance(config, DocDatasetConfig):
            self._field = "docs"
            self._iterator = self.ir_dataset.docs_iter
            self._sample_cls = DocSample
        else:
            raise ValueError("Invalid dataset configuration.")

    @property
    def dataset_id(self) -> str:
        return self.ir_dataset.dataset_id()

    @property
    def docs_dataset_id(self) -> str:
        return ir_datasets.docs_parent_id(self.dataset_id)

    def __len__(self) -> int:
        return (
            getattr(self.config, f"num_{self._field}")
            or getattr(self.ir_dataset, f"{self._field}_count")()
        )

    def __iter__(self) -> Iterator[QuerySample | DocSample]:
        start = self.rank
        stop = getattr(self.config, f"num_{self._field}") or None
        step = self.num_replicas
        for sample in islice(self._iterator(), start, stop, step):
            yield self._sample_cls.from_ir_dataset_sample(sample)


class QueryDataset(DataParallelIterableDataset):
    def __init__(self, query_dataset: str, config: QueryDatasetConfig) -> None:
        super().__init__(query_dataset, config)
        self.config: QueryDatasetConfig

    def __iter__(self) -> Iterator[QuerySample]:
        yield from super().__iter__()


class DocDataset(DataParallelIterableDataset):
    def __init__(self, doc_dataset: str, config: DocDatasetConfig) -> None:
        super().__init__(doc_dataset, config)
        self.config: DocDatasetConfig

    def __iter__(self) -> Iterator[DocSample]:
        yield from super().__iter__()


class IRDataset:
    def __init__(self, dataset: str) -> None:
        if dataset in DASHED_DATASET_MAP:
            dataset = DASHED_DATASET_MAP[dataset]
        self.dataset = dataset
        try:
            self.ir_dataset = ir_datasets.load(dataset)
        except KeyError:
            self.ir_dataset = None
        self._queries = None
        self._docs = None

    @property
    def queries(self) -> pd.Series:
        if self._queries is None:
            if self.ir_dataset is None:
                raise ValueError(
                    f"Unable to find dataset {self.dataset} in ir-datasets"
                )
            queries_iter = self.ir_dataset.queries_iter()
            self._queries = pd.Series(
                {query.query_id: query.default_text() for query in queries_iter},
                name="text",
            )
            self._queries.index.name = "query_id"
        return self._queries

    @property
    def docs(self) -> ir_datasets.indices.Docstore | Dict[str, GenericDoc]:
        if self._docs is None:
            if self.ir_dataset is None:
                raise ValueError(
                    f"Unable to find dataset {self.dataset} in ir-datasets"
                )
            self._docs = self.ir_dataset.docs_store()
        return self._docs

    @property
    def dataset_id(self) -> str:
        if self.ir_dataset is None:
            return self.dataset
        return self.ir_dataset.dataset_id()

    @property
    def docs_dataset_id(self) -> str:
        return ir_datasets.docs_parent_id(self.dataset_id)


class RunDataset(IRDataset, Dataset):
    def __init__(
        self,
        run_path: Path,
        config: RunDatasetConfig,
        stage: Literal["train", "validate", "predict"] = "train",
    ) -> None:
        super().__init__(
            run_path.name[: -len("".join(run_path.suffixes))].split("__")[-1]
        )
        if stage == "train" and config.targets is None:
            raise ValueError("Targets are required for training.")
        self.run_path = run_path
        self.config = config
        self.stage = stage
        self.depth = config.depth

        self.run = self.load_run()
        self.qrels = self.load_qrels()
        self.qrel_groups = None

        if self.qrels is not None:
            self.run = self.run.merge(
                self.qrels.add_prefix("relevance_", axis=1),
                on=["query_id", "doc_id"],
                how=(
                    "outer" if self._docs is None else "left"
                ),  # outer join if docs are from ir_datasets else only keep docs in run
            )
            self.qrel_groups = self.qrels.groupby("query_id")

        if self.stage == "train":
            num_docs_per_query = self.run.groupby("query_id").transform("size")
            self.run = self.run[num_docs_per_query >= config.sample_size]

        self.run = self.run.sort_values(["query_id", "rank"])
        self.run_groups = self.run.groupby("query_id")
        self.query_ids = list(self.run_groups.groups.keys())

        if self.run["rank"].max() < config.depth:
            warnings.warn("Depth is greater than the maximum rank in the run file.")
        if config.sampling_strategy == "top" and config.sample_size > config.depth:
            warnings.warn(
                "Sample size is greater than depth and top sampling strategy is used. "
                "This can cause documents to be sampled that are not contained "
                "in the run file, but that are present in the qrels."
            )

    def load_run(self) -> pd.DataFrame:
        if set((".tsv", ".run", ".csv")).intersection(self.run_path.suffixes):
            run = pd.read_csv(
                self.run_path,
                sep=r"\s+",
                header=None,
                names=RUN_HEADER,
                usecols=[0, 2, 3, 4],
                dtype={"query_id": str, "doc_id": str},
            )
        elif set((".json", ".jsonl")).intersection(self.run_path.suffixes):
            kwargs = {}
            if ".jsonl" in self.run_path.suffixes:
                kwargs["lines"] = True
                kwargs["orient"] = "records"
            run = pd.read_json(
                self.run_path,
                **kwargs,
                dtype={
                    "query_id": str,
                    "qid": str,
                    "doc_id": str,
                    "docid": str,
                    "docno": str,
                },
            ).rename(
                {
                    "qid": "query_id",
                    "docid": "doc_id",
                    "docno": "doc_id",
                },
                axis=1,
            )
            if "query" in run.columns:
                self._queries = (
                    run.drop_duplicates("query_id")
                    .set_index("query_id")["query"]
                    .rename("text")
                )
                run = run.drop("query", axis=1)
            if "text" in run.columns:
                self._docs = (
                    run.set_index("doc_id")["text"]
                    .map(lambda x: GenericDoc("", x))
                    .to_dict()
                )
                run = run.drop("text", axis=1)
        else:
            raise ValueError("Invalid run file format.")
        if self.depth != -1:
            run = run[run["rank"] <= self.config.depth]
        return run

    def load_qrels(self) -> pd.DataFrame | None:
        if self.stage == "predict":
            return None
        if self.ir_dataset is None:
            return None
        qrels = pd.DataFrame(self.ir_dataset.qrels_iter()).rename(
            {"subtopic_id": "iteration"}, axis=1
        )
        if "iteration" not in qrels.columns:
            qrels["iteration"] = 0
        qrels = qrels.drop_duplicates(["query_id", "doc_id", "iteration"])
        qrels = qrels.set_index(["query_id", "doc_id", "iteration"]).unstack(level=-1)
        qrels = qrels.droplevel(0, axis=1)
        qrels = qrels.loc[pd.IndexSlice[self.run["query_id"].drop_duplicates(), :]]
        return qrels

    def __len__(self) -> int:
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> RunSample:
        query_id = str(self.query_ids[idx])
        group = self.run_groups.get_group(query_id).copy()
        query = self.queries[query_id]
        if self.config.sampling_strategy == "single_relevant":
            relevant = group.loc[
                group.filter(like="relevance").max(axis=1).gt(0)
            ].sample(1)
            non_relevant_bool = (
                group.filter(like="relevance").max(axis=1).fillna(0).eq(0)
                & ~group["rank"].isna()
            )
            num_non_relevant = non_relevant_bool.sum()
            sample_non_relevant = min(self.config.sample_size - 1, num_non_relevant)
            non_relevant = group.loc[non_relevant_bool].sample(sample_non_relevant)
            group = pd.concat([relevant, non_relevant])
        elif self.config.sampling_strategy == "top":
            group = group.head(self.config.sample_size)
        else:
            raise ValueError("Invalid sampling strategy.")

        doc_ids = tuple(group["doc_id"])
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)

        targets = None
        if self.config.targets is not None:
            targets = torch.tensor(
                group.set_index("doc_id")
                .loc[list(doc_ids)]
                .filter(like=self.config.targets)
                .fillna(0)
                .values
            )
        qrels = None
        if self.qrel_groups is not None:
            qrels = (
                self.qrel_groups.get_group(query_id)
                .stack()
                .rename("relevance")
                .astype(int)
                .reset_index()
                .to_dict(orient="records")
            )
        return RunSample(query_id, query, doc_ids, docs, targets, qrels)


class TupleDataset(IRDataset, IterableDataset):
    def __init__(
        self,
        tuples_dataset: str,
        config: TupleDatasetConfig,
        stage: Literal["train", "validate"] = "train",
    ) -> None:
        super().__init__(tuples_dataset)
        if self.queries is None:
            raise ValueError("Queries are required for run datasets.")
        self.config = config
        self.targets = self.config.targets
        self.stage = stage
        if self.stage not in {"train", "validate"}:
            raise ValueError(
                f"Invalid stage. Expected 'train' or 'validate'. Got {stage}."
            )

    def parse_sample(
        self, sample: ScoredDocTuple | GenericDocPair
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[float, ...] | None]:
        targets = None
        if isinstance(sample, GenericDocPair):
            if self.targets == "score":
                raise ValueError("ScoredDocTuple required for score targets.")
            elif self.targets == "order":
                targets = (1.0, 0.0)
            doc_ids = (sample.doc_id_a, sample.doc_id_b)
        elif isinstance(sample, ScoredDocTuple):
            doc_ids = sample.doc_ids[: self.config.num_docs]
            if self.targets is not None:
                targets = (
                    sample.scores
                    if sample.scores is not None and self.targets == "score"
                    else tuple([1.0] + [0.0] * sample.num_docs)
                )
                targets = targets[: self.config.num_docs]
        else:
            raise ValueError("Invalid sample type.")
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)
        return doc_ids, docs, targets

    def __iter__(self) -> Iterator[RunSample]:
        for sample in self.ir_dataset.docpairs_iter():
            query_id = sample.query_id
            query = self.queries.loc[query_id]
            doc_ids, docs, targets = self.parse_sample(sample)
            if targets is not None:
                targets = torch.tensor(targets)
            yield RunSample(query_id, query, doc_ids, docs, targets)
