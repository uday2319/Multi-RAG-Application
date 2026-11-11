import math
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text):
    """
    Lowercase word tokens. Keeps hyphenated and dotted terms
    together, e.g. "rag-token", "bart-large", "78.4".
    """
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Keyword index over the same chunks that are in FAISS.
    Built in memory at startup from the vector store metadata.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi(
            [tokenize(chunk["content"]) for chunk in chunks]
        )

    def search(self, query):
        """Returns [(chunk_index, bm25_score), ...] for all matching chunks, best first."""

        scores = self.bm25.get_scores(tokenize(query))

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            (i, float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


class Retriever:

    def __init__(
        self,
        embedder,
        vectorstore,
        mode="dense",
        candidates=30,
        rrf_k=60,
        reranker_model=None,
        router=None,
        routing="off"
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k

        # "off": never route; "multi": route when 2+ papers are
        # named; "all": route whenever any paper is named
        self.router = router
        self.routing = routing if router else "off"

        self.bm25 = (
            BM25Index(vectorstore.metadata)
            if mode == "hybrid" else None
        )

        self.reranker = (
            CrossEncoder(reranker_model)
            if reranker_model else None
        )

    # -----------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------

    def rrf_fuse(self, *ranked_lists):
        """
        Reciprocal Rank Fusion: each list gives a chunk
        1 / (rrf_k + rank). Chunks found by both retrievers
        add up, so agreement is rewarded.
        """

        fused = {}

        for ranked in ranked_lists:
            for rank, (index, _) in enumerate(ranked, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (self.rrf_k + rank)

        return sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True
        )

    def get_candidates(self, query, query_embedding, sources=None):
        """
        Top candidates from dense (and BM25 in hybrid mode),
        optionally restricted to a set of source papers.
        Returns [(chunk_index, fusion_or_dense_score), ...].
        """

        metadata = self.vectorstore.metadata

        def allowed(index):
            return sources is None or metadata[index]["source"] in sources

        # With a source filter, search everything and filter;
        # the corpus is small enough for this to be cheap.
        dense_k = self.candidates if sources is None else len(metadata)

        dense = [
            hit for hit in self.vectorstore.search_ids(
                query_embedding, top_k=dense_k
            )
            if allowed(hit[0])
        ][:self.candidates]

        if not self.bm25:
            return dense, dict(dense)

        keyword = [
            hit for hit in self.bm25.search(query)
            if allowed(hit[0])
        ][:self.candidates]

        return self.rrf_fuse(dense, keyword)[:self.candidates], dict(dense)

    def to_results(self, query_embedding, candidates, dense_scores):
        """Turn (index, score) pairs into result dicts with dense_score."""

        results = []

        for index, score in candidates:

            item = self.vectorstore.metadata[index].copy()
            item["dense_score"] = dense_scores.get(index)
            item["score"] = float(score)
            results.append(item)

        # Chunks found only by BM25 have no cosine score yet;
        # compute it so every result has one (used by the guard)
        missing = [r for r in results if r["dense_score"] is None]

        if missing:
            vectors = self.embedder.embed([r["content"] for r in missing])
            for result, vector in zip(missing, vectors):
                result["dense_score"] = float(query_embedding[0] @ vector)

        return results

    def rerank(self, query, results):

        if not self.reranker or not results:
            return results

        scores = self.reranker.predict(
            [(query, r["content"]) for r in results]
        )

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            result["score"] = float(score)

        return sorted(
            results,
            key=lambda r: r["rerank_score"],
            reverse=True
        )

    def search(self, query, query_embedding, sources=None):

        candidates, dense_scores = self.get_candidates(
            query, query_embedding, sources
        )

        results = self.to_results(query_embedding, candidates, dense_scores)

        return self.rerank(query, results)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def named_papers(self, query):

        if self.routing == "off":
            return []

        papers = self.router.detect(query)

        if self.routing == "multi" and len(papers) < 2:
            return []

        return papers

    def retrieve(self, query, top_k=5):

        query_embedding = self.embedder.embed([query])

        papers = self.named_papers(query)

        if not papers:
            return self.search(query, query_embedding)[:top_k]

        # Paper routing: search inside each named paper, then
        # interleave (best of paper A, best of paper B, ...) so
        # every named paper appears in the first results.
        per_paper = math.ceil(top_k / len(papers))

        lists = [
            self.search(query, query_embedding, sources={paper})[:per_paper]
            for paper in papers
        ]

        interleaved = []

        for rank in range(per_paper):
            for results in lists:
                if rank < len(results):
                    interleaved.append(results[rank])

        return interleaved[:max(top_k, len(papers))]
