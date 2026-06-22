"""Compatibility shim for llama-index-vector-stores-chroma 0.2.x + chromadb 0.5.x.

When a vector query has no metadata filters, the 0.2.x ChromaVectorStore passes
``where={}`` to chromadb. chromadb 0.5.x rejects an empty filter with
``Expected where to have exactly one operator, got {}`` — so unfiltered
semantic search (the common case) fails. We coerce empty ``where`` /
``where_document`` to ``None`` (chromadb's "no filter") on the two call sites.
"""
import functools


def apply_chroma_empty_filter_fix() -> None:
    try:
        from llama_index.vector_stores.chroma import ChromaVectorStore as CVS
    except Exception:
        return

    if getattr(CVS, "_empty_where_patched", False):
        return

    def _wrap(orig):
        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            if not kwargs.get("where"):
                kwargs["where"] = None
            if not kwargs.get("where_document"):
                kwargs.pop("where_document", None)
            return orig(self, *args, **kwargs)

        return wrapper

    for name in ("_query", "_get"):
        orig = getattr(CVS, name, None)
        if orig is not None:
            setattr(CVS, name, _wrap(orig))

    CVS._empty_where_patched = True
