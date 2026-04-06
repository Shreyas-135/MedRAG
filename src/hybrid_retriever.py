"""
Hybrid Sparse-Dense Retrieval for Medical RAG

Combines BM25 sparse retrieval with ChromaDB dense retrieval using
Reciprocal Rank Fusion (RRF), aligning with the project goal of
privacy-preserving, verifiable, cross-hospital medical imaging.

Key design points:
- BM25 handles exact medical-term matching (e.g. "ground-glass opacity")
  that pure cosine similarity can miss for low-frequency clinical terms.
- RRF fuses the two ranked lists without requiring score normalisation.
- ``last_reranker_params`` is populated after every ``retrieve()`` call so
  it can be passed directly to ``provenance.hash_retrieval_params()`` to
  fill the previously-unused ``reranker_params`` slot, enabling
  blockchain-verifiable hybrid retrieval provenance.

Usage::

    from hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(knowledge_base)
    results = retriever.retrieve(
        query_embedding=embedding_512d,
        query_text="bilateral ground-glass opacities",
        top_k=5,
    )
    # Pass to provenance hashing:
    retrieval_hash = hash_retrieval_params(
        item_ids=[r["id"] for r in results],
        similarity_scores=[r["similarity"] for r in results],
        top_k=5,
        reranker_params=retriever.get_reranker_params(),
    )
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print(
        "Warning: rank-bm25 not installed. Hybrid retrieval will fall back to "
        "dense-only retrieval.  Install with:  pip install rank-bm25"
    )


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case, strip non-alphanumeric (except hyphens), split on whitespace.

    Hyphens are preserved so clinical terms like 'COVID-19', 'ground-glass',
    and 'X-ray' are not split into unrelated tokens.
    """
    return re.sub(r"[^a-zA-Z0-9\s\-]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> Dict[str, float]:
    """
    Compute Reciprocal Rank Fusion scores over multiple ranked lists.

    RRF(d) = Σ_r  1 / (k + rank_r(d))

    where rank_r(d) is the 1-based rank of document *d* in list *r*.
    Documents absent from a list receive no contribution from that list.

    Args:
        ranked_lists: Each inner list contains document IDs in ranked order
                      (best first).
        k: Smoothing constant (Cormack et al., 2009 recommend k=60).

    Returns:
        Mapping from doc_id to cumulative RRF score (higher is better).
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid sparse-dense retriever with Reciprocal Rank Fusion.

    Maintains a lazy BM25 index that is rebuilt whenever the underlying
    ChromaDB knowledge base changes (detected via ``get_hash()``).  Dense
    retrieval is delegated to the existing ``ChromaDBMedicalKnowledgeBase``
    so no data is duplicated.

    After every call to ``retrieve()``, the ``last_reranker_params``
    attribute is updated with the parameters used during that call.  Pass
    this dict to ``provenance.hash_retrieval_params(reranker_params=...)``
    to produce a blockchain-verifiable provenance hash that captures both
    the dense and sparse retrieval decisions.

    Args:
        knowledge_base: A ``ChromaDBMedicalKnowledgeBase`` instance.
        rrf_k: RRF smoothing constant (default 60).
        dense_candidate_k: Number of candidates fetched from each retriever
                           before RRF fusion.  Should be larger than the
                           final ``top_k`` to give RRF enough candidates to
                           re-rank from.
    """

    def __init__(
        self,
        knowledge_base,
        rrf_k: int = 60,
        dense_candidate_k: int = 50,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.rrf_k = rrf_k
        self.dense_candidate_k = dense_candidate_k

        # BM25 index (rebuilt on KB change)
        self._bm25: Optional["BM25Okapi"] = None
        self._indexed_docs: List[Dict[str, Any]] = []
        self._kb_hash_at_index: str = ""

        # Exposed for provenance anchoring — updated on every retrieve() call
        self.last_reranker_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _rebuild_index_if_needed(self) -> None:
        """Rebuild the BM25 index when the KB has changed."""
        if not BM25_AVAILABLE:
            return
        current_hash = self.knowledge_base.get_hash()
        if current_hash == self._kb_hash_at_index and self._bm25 is not None:
            return  # Index is already up-to-date

        all_docs = self.knowledge_base.get_all_texts()
        if not all_docs:
            self._bm25 = None
            self._indexed_docs = []
            self._kb_hash_at_index = current_hash
            return

        self._indexed_docs = all_docs
        tokenized = [_tokenize(d["text"]) for d in all_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._kb_hash_at_index = current_hash

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-*k* results using hybrid BM25+dense+RRF fusion.

        When BM25 is unavailable or the KB is empty the method falls back to
        pure dense retrieval so existing behaviour is always preserved.

        Args:
            query_embedding: Dense query vector (CNN backbone output).
            query_text: Free-text query used for BM25 scoring.  Typically
                        set to the predicted class label plus any clinical
                        finding keywords.
            top_k: Final number of results to return.
            filter_metadata: Optional ChromaDB metadata filter forwarded to
                             the dense retrieval step.

        Returns:
            List of result dicts, each containing:
            ``{id, text, similarity, metadata, rrf_score,
            retrieval_method}``.
        """
        self._rebuild_index_if_needed()
        candidate_k = max(self.dense_candidate_k, top_k * 10, 20)

        # ---- Dense retrieval ------------------------------------------------
        dense_results = self.knowledge_base.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filter_metadata=filter_metadata,
        )
        dense_id_to_result = {
            r.get("id", str(i)): r for i, r in enumerate(dense_results)
        }
        dense_ranked = [r.get("id", str(i)) for i, r in enumerate(dense_results)]

        # ---- Sparse (BM25) retrieval ----------------------------------------
        sparse_ranked: List[str] = []
        bm25_scores: Dict[str, float] = {}
        if BM25_AVAILABLE and self._bm25 is not None and query_text.strip():
            query_tokens = _tokenize(query_text)
            raw_scores = self._bm25.get_scores(query_tokens)
            ranked_indices = np.argsort(raw_scores)[::-1][:candidate_k]
            for idx in ranked_indices:
                doc_id = self._indexed_docs[idx]["id"]
                sparse_ranked.append(doc_id)
                bm25_scores[doc_id] = float(raw_scores[idx])

        # ---- RRF fusion -----------------------------------------------------
        if sparse_ranked:
            rrf_scores = _reciprocal_rank_fusion(
                [dense_ranked, sparse_ranked], k=self.rrf_k
            )
            retrieval_method = "hybrid_rrf"
        else:
            # Pure dense fallback: assign RRF-equivalent scores to preserve
            # the output dict format while noting the fallback.
            rrf_scores = {
                doc_id: 1.0 / (self.rrf_k + rank)
                for rank, doc_id in enumerate(dense_ranked, start=1)
            }
            retrieval_method = "dense_only"

        # ---- Build final ranked list ----------------------------------------
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True
        )
        final_ids = sorted_ids[:top_k]

        id_to_indexed = {d["id"]: d for d in self._indexed_docs}
        results: List[Dict[str, Any]] = []
        for doc_id in final_ids:
            if doc_id in dense_id_to_result:
                entry = dict(dense_id_to_result[doc_id])
            elif doc_id in id_to_indexed:
                # Only in sparse results — reconstruct a minimal result dict
                entry = {
                    "id": doc_id,
                    "text": id_to_indexed[doc_id]["text"],
                    "metadata": id_to_indexed[doc_id].get("metadata", {}),
                    "similarity": bm25_scores.get(doc_id, 0.0),
                }
            else:
                continue
            entry["rrf_score"] = rrf_scores[doc_id]
            entry["retrieval_method"] = retrieval_method
            results.append(entry)

        # ---- Record params for blockchain provenance ------------------------
        self.last_reranker_params = {
            "strategy": retrieval_method,
            "rrf_k": self.rrf_k,
            "bm25_available": BM25_AVAILABLE and self._bm25 is not None,
            "dense_candidate_k": candidate_k,
            "sparse_candidate_count": len(sparse_ranked),
            "query_text_hash": hashlib.sha256(query_text.encode()).hexdigest(),
        }

        return results

    def get_reranker_params(self) -> Dict[str, Any]:
        """Return reranker params from the last ``retrieve()`` call.

        Pass this to ``provenance.hash_retrieval_params(reranker_params=...)``
        to produce a blockchain-verifiable hash that captures the hybrid
        retrieval decision.
        """
        return dict(self.last_reranker_params)
