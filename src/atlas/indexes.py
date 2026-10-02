"""Derived indexes: BM25 (sparse), vectors (dense), and the temporal knowledge graph."""

from __future__ import annotations

import functools
import hashlib
import itertools
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

import numpy as np

from .config import Settings
from .domain import Chunk, Entity, Relation
from .ontology import SYMMETRIC
from .store import Repository

# ------------------------------------------------------------ tokenizer
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "s", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "about", "also", "more", "most", "other", "some", "such", "than", "very"])
_COMPOUND = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


@functools.lru_cache(maxsize=65536)
def stem(t: str) -> str:
    if len(t) <= 3 or t.isdigit() or any(c.isdigit() for c in t):
        return t
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if t.endswith(suf) and len(t) - len(suf) >= 3 and not (suf == "s" and t.endswith("ss")):
            return t[: len(t) - len(suf)] + rep
    return t


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """Lowercase, keep compound technical tokens ('gpt-4', 'llama-3.1', '8x7b') AND their parts and joined form."""
    out: list[str] = []
    for m in _COMPOUND.finditer(text.lower().replace("’", "'")):
        tok = m.group()
        parts = re.split(r"[-.]", tok)
        if len(parts) > 1:
            out.append(tok)
            out.append("".join(parts))
            out.extend(p for p in parts if p and (keep_stop or p not in STOPWORDS))
        elif keep_stop or tok not in STOPWORDS:
            out.append(tok)
    return [stem(t) for t in out]


# ----------------------------------------------------------------- BM25
class BM25Index:
    """Incremental inverted index with BM25 scoring; title/section terms get a field boost (BM25F-lite)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, field_boost: float = 2.0) -> None:
        self.k1, self.b, self.boost = k1, b, field_boost
        self.postings: dict[str, dict[str, float]] = defaultdict(dict)
        self.dl: dict[str, float] = {}
        self.total_len = 0.0

    def __len__(self) -> int:
        return len(self.dl)

    def add(self, c: Chunk) -> None:
        if c.chunk_id in self.dl:
            self.remove(c.chunk_id)
        tf: Counter[str] = Counter(tokenize(c.text))
        for t in tokenize(f"{c.title} {c.section_title}"):
            tf[t] += self.boost
        for t, f in tf.items():
            self.postings[t][c.chunk_id] = f
        self.dl[c.chunk_id] = sum(tf.values())
        self.total_len += self.dl[c.chunk_id]

    def remove(self, chunk_id: str) -> None:
        if chunk_id not in self.dl:
            return
        self.total_len -= self.dl.pop(chunk_id)
        for t in [t for t, p in self.postings.items() if chunk_id in p]:
            del self.postings[t][chunk_id]
            if not self.postings[t]:
                del self.postings[t]

    def idf(self, term: str) -> float:
        n, df = len(self.dl), len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def search(self, query: str, k: int, allow: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        if not self.dl:
            return []
        avg = self.total_len / len(self.dl)
        scores: dict[str, float] = defaultdict(float)
        for t in set(tokenize(query)):
            post = self.postings.get(t)
            if not post:
                continue
            idf = self.idf(t)
            for cid, f in post.items():
                if allow and not allow(cid):
                    continue
                scores[cid] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[cid] / avg))
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


# ------------------------------------------------------------ embedders
# Concept groups give the offline embedder a thin semantic layer: synonyms hash to a shared token.
CONCEPTS: dict[str, list[str]] = {
    "open": ["open-source", "open-weight", "open-weights", "opensource", "permissive", "apache", "mit license", "openly"],
    "moe": ["mixture-of-experts", "mixture of experts", "moe", "sparse experts", "experts", "mixtral"],
    "release": ["release", "released", "launch", "launched", "unveil", "unveiled", "announce", "announced", "introduce",
                "introduced", "debut", "debuted", "ship", "shipped", "publish", "published"],
    "chip": ["accelerator", "accelerators", "gpu", "gpus", "tpu", "chip", "chips", "silicon", "trainium", "maia", "blackwell", "instinct"],
    "partner": ["partner", "partnered", "partnership", "collaboration", "collaborate", "collaborated", "alliance", "teamed"],
    "model": ["model", "models", "llm", "llms", "language model", "foundation model"],
    "multimodal": ["multimodal", "vision", "image", "audio", "video", "omni", "natively multimodal"],
    "reasoning": ["reasoning", "chain-of-thought", "reinforcement", "think", "thinking"],
    "retrieval": ["retrieval", "rag", "retrieve", "grounding", "search"],
    "attention": ["attention", "transformer", "transformers", "self-attention"],
    "efficient": ["efficient", "efficiency", "cheaper", "lora", "low-rank", "parameter-efficient", "quantization"],
    "context": ["context window", "context length", "long context", "tokens of context"],
}
_CONCEPT_LOOKUP: dict[str, str] = {}
for _c, _ws in CONCEPTS.items():
    for _w in _ws:
        _CONCEPT_LOOKUP[_w] = _c


class Embedder:
    name = "base"
    dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalised
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic, offline feature-hashing embedder: sublinear-tf unigrams+bigrams plus concept tokens.
    A lexical/concept baseline standing in for a neural model; swap via ATLAS_EMBEDDER=st:<model>."""

    def __init__(self, dim: int = 512) -> None:
        self.dim, self.name = dim, f"hashing-{dim}"

    @staticmethod
    def _h(tok: str) -> int:
        return int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")

    def _features(self, text: str) -> Counter[str]:
        low = text.lower()
        toks = tokenize(text)
        f: Counter[str] = Counter(toks)
        for a, b in itertools.pairwise(toks):
            f[f"{a}_{b}"] += 0.7
        for phrase, concept in _CONCEPT_LOOKUP.items():
            if (f" {phrase} " in f" {low} ") or (phrase in low and len(phrase) > 5):
                f[f"__c_{concept}"] += 1.5
        return f

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok, w in self._features(t).items():
                h = self._h(tok)
                out[i, h % self.dim] += (1.0 if (h >> 33) & 1 else -1.0) * (1 + math.log(w) if w >= 1 else w)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model)
        self.name = f"st-{model}"
        self.dim = int(self.m.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def make_embedder(s: Settings) -> Embedder:
    if s.embedder.startswith("st:"):
        return SentenceTransformerEmbedder(s.embedder[3:])
    return HashingEmbedder(s.embedding_dim)


class VectorIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.vecs: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.vecs)

    def add(self, chunk_id: str, vec: np.ndarray) -> None:
        self.vecs[chunk_id] = vec
        self._mat = None

    def remove(self, chunk_id: str) -> None:
        if self.vecs.pop(chunk_id, None) is not None:
            self._mat = None

    def _matrix(self) -> np.ndarray:
        if self._mat is None:
            self._ids = list(self.vecs)
            self._mat = np.vstack([self.vecs[i] for i in self._ids]) if self._ids else np.zeros((0, self.embedder.dim), np.float32)
        return self._mat

    def search_vec(self, q: np.ndarray, k: int, allow: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        m = self._matrix()
        if not len(m):
            return []
        sims = m @ q
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for j in order:
            cid = self._ids[int(j)]
            if allow and not allow(cid):
                continue
            out.append((cid, float(sims[j])))
            if len(out) >= k:
                break
        return out

    def search(self, query: str, k: int, allow: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        return self.search_vec(self.embedder.embed([query])[0], k, allow)


# ---------------------------------------------------------------- graph
class KnowledgeGraph:
    """Directed multigraph over (entity)-[relation]->(entity) with validity intervals.
    Symmetric relations are traversable in both directions."""

    def __init__(self) -> None:
        self.nodes: dict[str, Entity] = {}
        self.rels: dict[str, Relation] = {}
        self.out: dict[str, list[str]] = defaultdict(list)
        self.inc: dict[str, list[str]] = defaultdict(list)

    def load(self, entities: Iterable[Entity], relations: Iterable[Relation]) -> None:
        self.__init__()  # type: ignore[misc]
        for e in entities:
            self.nodes[e.entity_id] = e
        for r in relations:
            self.rels[r.relation_id] = r
            self.out[r.src_id].append(r.relation_id)
            self.inc[r.dst_id].append(r.relation_id)

    def edges(self, entity_id: str, direction: str = "both", rel: str | set[str] | None = None,
              active_only: bool = False) -> list[tuple[Relation, str]]:
        """[(relation, other_entity_id)] touching entity_id."""
        want = {rel} if isinstance(rel, str) else rel
        out: list[tuple[Relation, str]] = []
        if direction in ("out", "both"):
            out += [(self.rels[i], self.rels[i].dst_id) for i in self.out.get(entity_id, ())]
        if direction in ("in", "both"):
            out += [(self.rels[i], self.rels[i].src_id) for i in self.inc.get(entity_id, ())]
        if direction == "out":  # symmetric relations also reachable "outwards" from the dst side
            out += [(self.rels[i], self.rels[i].src_id) for i in self.inc.get(entity_id, ()) if self.rels[i].rel in SYMMETRIC]
        if direction == "in":
            out += [(self.rels[i], self.rels[i].dst_id) for i in self.out.get(entity_id, ()) if self.rels[i].rel in SYMMETRIC]
        seen, res = set(), []
        for r, o in out:
            if (want and r.rel not in want) or (active_only and not r.active) or (r.relation_id, o) in seen:
                continue
            seen.add((r.relation_id, o))
            res.append((r, o))
        return res

    def history(self, entity_id: str) -> list[Relation]:
        """All relations touching the entity ordered by when they became valid (the entity's timeline)."""
        ids = {*self.out.get(entity_id, ()), *self.inc.get(entity_id, ())}
        return sorted((self.rels[i] for i in ids), key=lambda r: (r.valid_from or r.observed_at or "", r.rel))

    def neighborhood(self, seeds: Iterable[str], hops: int = 2, active_only: bool = False) -> dict[str, tuple[int, list[Relation]]]:
        """BFS: entity_id -> (distance, relations on the path edge that reached it)."""
        dist: dict[str, tuple[int, list[Relation]]] = {s: (0, []) for s in seeds if s in self.nodes}
        frontier = list(dist)
        for d in range(1, hops + 1):
            nxt = []
            for e in frontier:
                for r, o in self.edges(e, active_only=active_only):
                    if o not in dist:
                        dist[o] = (d, [r])
                        nxt.append(o)
            frontier = nxt
        return dist

    def path(self, a: str, b: str, max_hops: int = 3) -> list[Relation] | None:
        if a == b:
            return []
        prev: dict[str, tuple[str, Relation]] = {}
        frontier, seen = [a], {a}
        for _ in range(max_hops):
            nxt = []
            for e in frontier:
                for r, o in self.edges(e):
                    if o in seen:
                        continue
                    seen.add(o)
                    prev[o] = (e, r)
                    if o == b:
                        path, cur = [], b
                        while cur != a:
                            p, rr = prev[cur]
                            path.append(rr)
                            cur = p
                        return path[::-1]
                    nxt.append(o)
            frontier = nxt
        return None


# --------------------------------------------------------- index manager
class IndexManager:
    """Owns all derived indexes and the in-memory chunk table; rebuilt from the repository on load."""

    def __init__(self, repo: Repository, settings: Settings, embedder: Embedder | None = None) -> None:
        self.repo, self.settings = repo, settings
        self.embedder = embedder or make_embedder(settings)
        self.bm25, self.vectors, self.graph = BM25Index(), VectorIndex(self.embedder), KnowledgeGraph()
        self.chunks: dict[str, Chunk] = {}
        self.version = 0  # bumps on every mutation; invalidates caches

    def load(self) -> None:
        self.bm25, self.vectors, self.graph = BM25Index(), VectorIndex(self.embedder), KnowledgeGraph()
        self.chunks = {c.chunk_id: c for c in self.repo.all_active_chunks()}
        stored = self.repo.vectors(self.embedder.name)
        missing = []
        for c in self.chunks.values():
            self.bm25.add(c)
            if c.chunk_id in stored:
                self.vectors.add(c.chunk_id, stored[c.chunk_id])
            else:
                missing.append(c)
        self._embed_and_store(missing)
        self.reload_graph()
        self.version += 1

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        for i in range(0, len(chunks), 64):
            batch = chunks[i:i + 64]
            mat = self.embedder.embed([c.contextual_text for c in batch])
            for c, v in zip(batch, mat):
                self.repo.put_vector(c.chunk_id, self.embedder.name, v)
                self.vectors.add(c.chunk_id, v)
        return len(chunks)

    def reload_graph(self) -> None:
        self.graph.load(self.repo.entities(), self.repo.relations())

    def apply(self, new: list[Chunk], refreshed: list[Chunk], removed: list[str]) -> int:
        """Incremental update. Only *new* chunks are embedded. Returns number embedded."""
        for cid in removed:
            self.bm25.remove(cid)
            self.vectors.remove(cid)
            self.chunks.pop(cid, None)
        for c in refreshed:
            self.chunks[c.chunk_id] = c
            self.bm25.add(c)
        n = self._embed_and_store(new)
        for c in new:
            self.chunks[c.chunk_id] = c
            self.bm25.add(c)
        self.reload_graph()
        self.version += 1
        return n
