import json
from pathlib import Path

import faiss
import numpy as np


class FAISSStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add(self, embeddings, metadata):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings and metadata items must match."
            )

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search_ids(self, query_embedding, top_k=5):
        """
        Like search(), but returns [(chunk_index, score), ...]
        so results from different retrievers can be fused.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        return [
            (int(index), float(score))
            for score, index in zip(scores[0], indices[0])
            if index != -1
        ]

    def search(self, query_embedding, top_k=5):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            item = self.metadata[index].copy()
            item["score"] = float(score)

            results.append(item)

        return results

    def save(self, path):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss")
        )

        with open(
            path / "metadata.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path):

        path = Path(path)

        self.index = faiss.read_index(
            str(path / "index.faiss")
        )
        self.dimension = self.index.d

        with open(
            path / "metadata.json",
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)