"""A signal to compute semantic search for a document."""
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
from typing_extensions import override  # type: ignore

from ..embeddings.embedding_index import EmbeddingIndex
from ..embeddings.embedding_registry import EmbeddingId, EmbedFn
from ..schema import DataType, EnrichmentType, Field, Item, RichData
from .signal import Signal

SIMILARITY_FEATURE_NAME = 'semantic_similarity'


class SemanticSearchSignal(Signal):
  """A signal to compute semantic search for a document."""

  name = 'semantic_search'
  enrichment_type = EnrichmentType.TEXT
  embedding_based = True

  query: Union[str, bytes]
  embedding: EmbeddingId

  _embed_fn: EmbedFn
  _search_text_embedding: Optional[np.ndarray] = None

  def __init__(self, query: Union[str, bytes], embedding: EmbeddingId, **kwargs: dict[Any, Any]):
    if isinstance(query, bytes):
      raise ValueError('Image queries are not yet supported for SemanticSearch.')

    super().__init__(query=query, embedding=embedding, **kwargs)

  @override
  def fields(self) -> dict[str, Field]:
    return {SIMILARITY_FEATURE_NAME: Field(dtype=DataType.FLOAT32)}

  def _get_search_embedding(self) -> np.ndarray:
    """Return the embedding for the search text."""
    if self._search_text_embedding is None:
      self._search_text_embedding = self._embed_fn([self.query]).flatten()
    return self._search_text_embedding

  @override
  def compute(
      self,
      data: Optional[Iterable[RichData]] = None,
      keys: Optional[Iterable[bytes]] = None,
      get_embedding_index: Optional[Callable[[EmbeddingId, Iterable[bytes]], EmbeddingIndex]] = None
  ) -> Iterable[Optional[Item]]:
    if data and keys:
      raise ValueError('"data" and "keys" cannot both be provided for SemanticSearch.compute().')

    if data:
      text_embeddings = self._embed_fn(data)
    elif keys:
      if not get_embedding_index:
        raise ValueError(
            '"get_embedding_index" is required for SemanticSearch.compute() when passing "keys".')
      text_embeddings = get_embedding_index(self.embedding, keys).embeddings

    scores = text_embeddings.dot(self._get_search_embedding()).flatten()

    return [{SIMILARITY_FEATURE_NAME: score} for score in scores]
