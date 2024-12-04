import gc
import functools
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, cast

from typing_extensions import override
from ..splitters.chunk_splitter import TextChunk
from ..utils import log

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

from ..schema import Item
from ..signal import TextEmbeddingSignal
from ..splitters.spacy_splitter import clustering_spacy_chunker
from ..tasks import TaskExecutionType
from .embedding import chunked_compute_embedding, identity_chunker
from .transformer_utils import SENTENCE_TRANSFORMER_BATCH_SIZE

# See https://huggingface.co/spaces/mteb/leaderboard for leaderboard of models.
BGE_M3 = 'BAAI/bge-m3'

@functools.cache
def _get_and_cache_bge_m3(model_name: str) -> 'BGEM3FlagModel':
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        raise ImportError(
            'Could not import the "FlagEmbedding" python package. '
            'Please install it with `pip install "lilac[bge]".'
        )
    
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # Handle the missing 'device' attribute gracefully
    device = getattr(model, 'device', None)
    if device:
        log(f'[{model_name}] Using device:', device)
    else:
        log(f'[{model_name}] Device information not available. Check the model configuration.')
    
    return model


class BGEM3(TextEmbeddingSignal):
    """Computes BGE-M3 embeddings.

    <br>This embedding runs on-device. See the [model card](https://huggingface.co/BAAI/bge-m3)
    for details.
    """

    name: ClassVar[str] = 'bge-m3'
    display_name: ClassVar[str] = 'BGE-M3'
    local_batch_size: ClassVar[int] = SENTENCE_TRANSFORMER_BATCH_SIZE
    local_parallelism: ClassVar[int] = 1
    local_strategy: ClassVar[TaskExecutionType] = 'threads'
    supports_garden: ClassVar[bool] = False

    _model_name = BGE_M3
    _model: 'BGEM3FlagModel'

    @override
    def setup(self) -> None:
        self._model = _get_and_cache_bge_m3(self._model_name)

    @override
    def compute(self, docs: list[str]) -> list[Optional[Item]]:
        """Call the embedding function."""
        chunker = cast(
            Callable[[str], list[TextChunk]],
            clustering_spacy_chunker if self._split else identity_chunker,
        )
        return chunked_compute_embedding(
            lambda docs: self._model.encode(docs)['dense_vecs'],
            docs,
            self.local_batch_size * 16,
            chunker=chunker,
        )

    @override
    def teardown(self) -> None:
        if not hasattr(self, '_model'):
            return

        del self._model
        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
