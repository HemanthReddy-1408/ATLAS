"""Query intelligence: understanding, classification, rewriting/expansion, decomposition, HyDE, routing, planning."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .clock import Clock
from .domain import EntityType, QueryType
from .indexes import STOPWORDS
from .llm import LLM, LLMError
from .process import EntityResolver
from .retrieve import Filters, RetrievalConfig, Variant


# ---------------------------------------------------------------- time
@dataclass
class TimeConstraint:
    start: date | None = None
    end: date | None = None
    recency: bool = False
    text: str = ""

    @property
    def hard(self) -> bool:
        return self.start is not None or self.end is not None

    def describe(self) -> str:
        if self.hard:
            return f"{self.start or '…'} → {self.end or 'now'}"
        return "recent (soft preference)" if self.recency else "none"


_YEAR = r"((?:19|20)\d{2})"
RECENCY_RE = re.compile(r"\b(latest|newest|most recent|recent(ly)?|current(ly)?|right now|nowadays|today|state[- ]of[- ]the[- ]art|up[- ]to[- ]date)\b", re.I)


def parse_time(q: str, today: date) -> TimeConstraint:
    ql = q.lower()
    t = TimeConstraint(recency=bool(RECENCY_RE.search(q)))
    if m := re.search(rf"\b(?:between|from)\s+{_YEAR}\s+(?:and|to|through|until|-)\s+{_YEAR}\b", ql):
        t.start, t.end, t.text = date(int(m[1]), 1, 1), date(int(m[2]), 12, 31), m[0]
    elif m := re.search(rf"\bsince\s+{_YEAR}\b", ql):
        t.start, t.text = date(int(m[1]), 1, 1), m[0]
    elif m := re.search(rf"\b(?:after)\s+{_YEAR}\b", ql):
        t.start, t.text = date(int(m[1]) + 1, 1, 1), m[0]
    elif m := re.search(rf"\b(?:before|until|prior to)\s+{_YEAR}\b", ql):
        t.end, t.text = date(int(m[1]) - 1, 12, 31) if "before" in m[0] or "prior" in m[0] else date(int(m[1]), 12, 31), m[0]
    elif m := re.search(rf"\b(?:in|during|of|throughout)\s+{_YEAR}\b", ql):
        t.start, t.end, t.text = date(int(m[1]), 1, 1), date(int(m[1]), 12, 31), m[0]
    elif m := re.search(r"\b(?:over\s+)?the\s+(?:past|last)\s+(\d+|a|one|two|three)\s+(year|month)s?\b", ql):
        n = {"a": 1, "one": 1, "two": 2, "three": 3}.get(m[1]) or int(m[1])
        t.start, t.end, t.text = today - timedelta(days=(365 if m[2] == "year" else 30) * n), today, m[0]
    elif m := re.search(r"\b(last|this)\s+year\b", ql):
        y = today.year - (1 if m[1] == "last" else 0)
        t.start, t.end, t.text = date(y, 1, 1), date(y, 12, 31), m[0]
    return t


# --------------------------------------------------------- understanding
@dataclass
class QueryAnalysis:
    original: str
    entity_ids: list[str]
    entity_names: list[str]
    entity_types: dict[str, str]
    time: TimeConstraint
    qtype: QueryType
    secondary: list[QueryType]
    operations: list[str]
    keywords: list[str]
    rel_hints: set[str]
    clauses: list[str]
    complex: bool
    technical: bool
    wants_official: bool
    topic: str = ""

    def summary(self) -> dict:
        return {
            "type": self.qtype.value, "secondary": [s.value for s in self.secondary], "entities": self.entity_names,
            "time": self.time.describe(), "operations": self.operations, "keywords": self.keywords,
            "relation_hints": sorted(self.rel_hints), "complex": self.complex, "clauses": self.clauses,
        }


_REL_CUES: list[tuple[str, str]] = [
    (r"\b(released?|launch(ed)?|introduc(ed|e)|unveil(ed)?|announc(ed|e)|shipp?(ed)?|publish(ed)?)\b", "RELEASED"),
    (r"\b(develop(ed|s|ing)?|built|builds?|building|makes?|creat(ed|es?)|design(ed|s)?)\b", "DEVELOPS"),
    (r"\b(acquir(ed|es?)|bought|purchas(ed|es?))\b", "ACQUIRED"),
    (r"\b(partner(s|ed|ing|ship)?|collaborat\w*|alliance|working with)\b", "PARTNERS_WITH"),
    (r"\b(invest(ed|s|ment|ing)?|backed|funded)\b", "INVESTED_IN"),
    (r"\b(uses?|using|based on|built on|architecture|leverag\w+|adopt\w*)\b", "USES"),
    (r"\b(outperform\w*|beats?|surpass\w*|better than)\b", "OUTPERFORMS"),
    (r"\b(trained on|training data|dataset)\b", "TRAINED_ON"),
    (r"\b(benchmark\w*|evaluat\w+|score\w*)\b", "EVALUATED_ON"),
    (r"\b(introduced|proposed|paper)\b", "INTRODUCED"),
    (r"\b(accelerators?|chips?|gpus?|hardware|silicon)\b", "IS_A"),
]
_OPS = [
    ("comparison", r"\b(compare|comparison|versus|vs\.?|differences? between|differ|better than)\b"),
    ("evolution", r"\b(how (has|have|did)|evolv\w+|over time|changed?|trend\w*|progress\w*|timeline|history|since \d{4})\b"),
    ("identify_entities", r"\b(which|what|who) (?:[\w-]+ ){0,2}(companies|company|organi[sz]ations|labs|vendors|models|chips|people|players|products|tools|papers)\b|\bwho (developed|built|made|created|released|owns|founded|acquired)\b"),
    ("date_lookup", r"\bwhen (did|was|were|is)\b|\bwhat year\b|\bwhich year\b"),
    ("explain", r"\b(why|how does|how do|explain|what is|what are|describe|what does)\b"),
    ("enumerate", r"\b(list|all the|what are the|name the|which)\b"),
    ("architecture", r"\b(architectur\w*|design|mechanism|attention|experts?)\b"),
]
_HOP = re.compile(r"\b(and also|but also|that also|who also|which also|and (?:are|is|have|has) (?:also )?(?:building|developing|making|releasing)|whose|that (?:are|is) (?:also )?)\b", re.I)
_SPLIT = re.compile(r",?\s+and\s+(?=(?:which|what|who|how|when|why|where)\b)|,\s*(?=(?:which|what|who|how|when|why)\b)|;\s*|\?\s+", re.I)
_PRON = re.compile(r"\b(it|its|them|their|they|those|these|this)\b", re.I)
_QWORDS = re.compile(r"^(?:please\s+|can you\s+|could you\s+|tell me\s+|explain\s+)?(?:what(?:'s| is| are| was| were)?|which|who|when|how|why|where)\b\s*(?:is|are|was|were|has|have|did|does|do|the)?\s*", re.I)


class QueryUnderstanding:
    def __init__(self, resolver: EntityResolver, clock: Clock) -> None:
        self.res, self.clock = resolver, clock

    def analyze(self, q: str) -> QueryAnalysis:
        q = q.strip()
        ql = q.lower()
        ments = self.res.extract(q, discover=False, lenient=True)
        ids = list(dict.fromkeys(m.entity_id for m in ments))
        names, types = [], {}
        for i in ids:
            e = self.res.repo.entity(i)
            if e:
                names.append(e.canonical_name)
                types[i] = e.type
        tc = parse_time(q, self.clock.now().date())
        ops = [name for name, rx in _OPS if re.search(rx, ql)]
        hints = {rel for rx, rel in _REL_CUES if re.search(rx, ql)}
        clauses = [c.strip(" ,?") for c in _SPLIT.split(q) if c and c.strip(" ,?")]
        clauses = [c for c in clauses if len(c.split()) >= 3] or [q.strip(" ?")]
        technical = bool(re.search(r"\b\d+(?:\.\d+)?[bkmt]\b|\b\d+x\d+[bkmt]?\b|\b[a-z]+-?\d+(?:\.\d+)?\b", ql))

        matches: dict[QueryType, bool] = {
            QueryType.COMPARATIVE: "comparison" in ops or (len(ids) >= 2 and bool(re.search(r"\b(vs|versus|compare|or)\b", ql))),
            QueryType.TEMPORAL: ("evolution" in ops or (tc.hard and "date_lookup" not in ops) or bool(re.search(r"\bsince\b|\bover time\b", ql))),
            QueryType.RELATIONAL: (bool(hints & {"RELEASED", "DEVELOPS", "ACQUIRED", "PARTNERS_WITH", "INVESTED_IN"}) and (
                "identify_entities" in ops or bool(ids)) and "explain" not in ops) or "identify_entities" in ops,
            QueryType.ANALYTICAL: bool(re.search(r"\b(why|impact|implications?|trade-?offs?|advantages?|disadvantages?|pros|cons|how does|how do|influence|effect|trends?)\b", ql)),
            QueryType.EXPLORATORY: bool(re.search(r"\b(overview|landscape|ecosystem|tell me about|survey|state of|big picture|summari[sz]e)\b", ql)),
        }
        matches[QueryType.MULTI_HOP] = matches[QueryType.RELATIONAL] and bool(_HOP.search(q)) and len(ids) >= 1
        order = [QueryType.COMPARATIVE, QueryType.MULTI_HOP, QueryType.TEMPORAL, QueryType.RELATIONAL,
                 QueryType.ANALYTICAL, QueryType.EXPLORATORY]
        hit = [t for t in order if matches.get(t)]
        primary = hit[0] if hit else QueryType.FACTUAL
        secondary = [t for t in hit[1:]]
        # a date-lookup of a single fact is FACTUAL even though it mentions time
        if "date_lookup" in ops and primary in (QueryType.TEMPORAL, QueryType.RELATIONAL) and len(clauses) == 1:
            primary, secondary = QueryType.FACTUAL, [primary]
        kw = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-+]*", q) if w.lower() not in STOPWORDS and w.lower() not in
              {"compare", "tell", "explain", "please", "difference", "differences", "between", "changed", "change", "since", "latest"}]
        topic = self._topic(clauses[0], ids)
        return QueryAnalysis(q, ids, names, types, tc, primary, secondary, ops, kw, hints, clauses,
                             complex=len(clauses) > 1 or primary == QueryType.MULTI_HOP,
                             technical=technical, wants_official=bool(re.search(r"\b(official|announce\w*|press release|documentation|docs)\b", ql)),
                             topic=topic)

    @staticmethod
    def _topic(clause: str, ids: list[str]) -> str:
        c = _QWORDS.sub("", clause.strip(" ?"))
        c = re.sub(r"\b(has|have|had|changed|evolved|since \d{4}|in \d{4}|over time)\b", "", c, flags=re.I)
        return re.sub(r"\s+", " ", c).strip(" ,")


# ------------------------------------------------ rewrite / expand / hyde
EXPANSIONS: dict[str, list[str]] = {
    "llm": ["large language model", "foundation model"], "llms": ["large language models"],
    "open-source": ["open-weight", "open weights", "openly available"], "open source": ["open-weight", "open weights"],
    "rag": ["retrieval-augmented generation"], "moe": ["mixture of experts"], "gpu": ["accelerator"],
    "accelerator": ["AI chip", "GPU", "TPU"], "accelerators": ["AI chips", "GPUs"],
    "released": ["introduced", "announced", "launched"], "release": ["introduce", "announce", "launch"],
    "partner": ["collaborate", "partnership"], "partnering": ["collaborating", "partnership"],
    "architecture": ["design", "attention", "experts"], "fine-tuning": ["LoRA", "parameter-efficient"],
}


def rewrite(a: QueryAnalysis) -> str:
    """Retrieval-friendly form: question scaffolding removed, entity aliases normalised to canonical names."""
    text = a.original
    body = _QWORDS.sub("", text.strip(" ?"))
    body = re.sub(r"\b(please|can you|could you|tell me|i want to know|has|have|did|does|do)\b", " ", body, flags=re.I)
    for name in a.entity_names:  # canonical names guarantee the exact-term match BM25 relies on
        if name.lower() not in body.lower():
            body += f" {name}"
    return re.sub(r"\s+", " ", body).strip(" ,.?")


def expand(text: str, limit: int = 2) -> list[str]:
    low = text.lower()
    out = []
    for term, alts in EXPANSIONS.items():
        if re.search(rf"\b{re.escape(term)}\b", low):
            out.extend(alts)
    return list(dict.fromkeys(out))[: limit * 3]


def hyde_template(a: QueryAnalysis) -> str:
    """No-LLM stand-in for a hypothetical answer: declarative sentence with the entities and cue verbs."""
    verbs = {"RELEASED": "released", "DEVELOPS": "developed", "PARTNERS_WITH": "partnered with", "USES": "uses",
             "ACQUIRED": "acquired", "OUTPERFORMS": "outperforms", "INVESTED_IN": "invested in"}
    v = [verbs[h] for h in sorted(a.rel_hints) if h in verbs] or ["announced"]
    subj = " and ".join(a.entity_names[:3]) or a.topic or " ".join(a.keywords[:4])
    return f"{subj} {v[0]} {a.topic or ' '.join(a.keywords[:6])}. {' '.join(a.keywords[:8])}."


def hyde_llm(a: QueryAnalysis, llm: LLM) -> str | None:
    try:
        return llm.complete(
            "You write short, factual, encyclopedic passages about AI and technology. No preamble.",
            f"Write a 3-sentence passage that would directly answer this question, as it might appear in an official "
            f"announcement or technical blog. Question: {a.original}", max_tokens=160, temperature=0.2, fast=True).strip()
    except LLMError:
        return None


# -------------------------------------------------------- decomposition
@dataclass
class SubQuestion:
    step_id: str
    text: str
    kind: str = "main"  # main | clause | entity | window
    entity_ids: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    label: str = ""


def decompose(a: QueryAnalysis, resolver: EntityResolver, llm: LLM | None = None) -> list[SubQuestion]:
    if llm and a.complex:
        try:
            d = llm.complete_json(
                "You split complex research questions into 2-5 self-contained retrieval sub-questions. "
                'Return JSON: {"sub_questions": ["..."]}. Each must be understandable alone (no pronouns).',
                a.original, max_tokens=300, fast=True)
            subs = [s for s in (d.get("sub_questions") if isinstance(d, dict) else d) or [] if isinstance(s, str) and s.strip()]
            if 1 < len(subs) <= 6:
                return [SubQuestion(f"s{i + 1}", s, "clause", _ids(s, resolver)) for i, s in enumerate(subs)]
        except (LLMError, AttributeError):
            pass
    subs: list[SubQuestion] = []
    topic = a.topic
    if a.qtype == QueryType.COMPARATIVE and len(a.entity_ids) >= 2:
        name_tokens = {t.lower() for nm in a.entity_names for t in re.findall(r"[A-Za-z0-9.\-+]+", nm)}
        rest = " ".join(k for k in a.keywords if k.lower() not in name_tokens and k.lower() not in {"and", "or", "vs"})
        for n, (eid, name) in enumerate(zip(a.entity_ids, a.entity_names), 1):
            subs.append(SubQuestion(f"s{n}", f"{name} {rest or 'release architecture parameters benchmarks license'}", "entity", [eid], label=name))
    elif len(a.clauses) > 1:
        for i, c in enumerate(a.clauses, 1):
            if i > 1 and _PRON.search(c) and topic:
                c = f"{c} ({topic})"
            subs.append(SubQuestion(f"s{i}", c, "clause", _ids(c, resolver) or a.entity_ids[:1]))
    else:
        subs.append(SubQuestion("s1", a.original, "main", a.entity_ids))
    # evolution over a multi-year hard range => time windows, so every era is represented in the evidence
    if a.time.hard and a.time.start and (a.qtype == QueryType.TEMPORAL or "evolution" in a.operations):
        end = a.time.end or date.today()
        years = end.year - a.time.start.year
        if years >= 1:
            edges = sorted({a.time.start.year, a.time.start.year + years // 2 + 1, end.year + 1})
            wins = list(itertools.pairwise(edges))
            base = subs[0].text if len(subs) == 1 else (topic or a.original)
            n0 = len(subs)
            for j, (y0, y1) in enumerate(wins, 1):
                subs.append(SubQuestion(f"s{n0 + j}", f"{base} {y0}" + (f"-{y1 - 1}" if y1 - 1 > y0 else ""), "window",
                                        a.entity_ids[:2], f"{y0}-01-01", f"{y1 - 1}-12-31"))
    return subs


def _ids(text: str, resolver: EntityResolver) -> list[str]:
    return list(dict.fromkeys(m.entity_id for m in resolver.extract(text, discover=False, lenient=True)))


# ----------------------------------------------------------------- router
def route(a: QueryAnalysis, llm_available: bool = False) -> RetrievalConfig:
    """Query type -> retrieval strategy. Deterministic and inspectable; the agent may override on refinement."""
    t = a.qtype
    w = {"bm25": 1.0, "dense": 1.0, "graph": 0.5 if a.entity_ids else 0.0}
    cfg = RetrievalConfig(weights=w)
    why = []
    if t == QueryType.RELATIONAL:
        cfg.weights = {"bm25": 1.0, "dense": 0.6, "graph": 1.5 if a.entity_ids else 0.0}
        why.append("relational → graph-first + BM25")
    elif t == QueryType.MULTI_HOP:
        cfg.weights = {"bm25": 1.0, "dense": 0.8, "graph": 1.6}
        cfg.decompose, cfg.graph_hops = True, 2
        why.append("multi-hop → decompose + graph bridge nodes")
    elif t == QueryType.TEMPORAL:
        cfg.weights = {"bm25": 1.0, "dense": 1.0, "graph": 0.9 if a.entity_ids else 0.3}
        cfg.decompose, cfg.diversity = True, 0.35
        cfg.graph_active_only = False  # history includes retired facts
        why.append("temporal → time-windowed sub-queries, historic graph facts")
    elif t == QueryType.COMPARATIVE:
        cfg.weights = {"bm25": 1.0, "dense": 1.0, "graph": 1.0}
        cfg.decompose, cfg.diversity = True, 0.35
        why.append("comparative → one sub-query per entity")
    elif t == QueryType.EXPLORATORY:
        cfg.weights = {"bm25": 0.7, "dense": 1.5, "graph": 0.7 if a.entity_ids else 0.0}
        cfg.hyde, cfg.diversity = True, 0.5
        why.append("exploratory → dense-heavy, HyDE, high diversity")
    elif t == QueryType.ANALYTICAL:
        cfg.weights = {"bm25": 1.0, "dense": 1.2, "graph": 0.5 if a.entity_ids else 0.0}
        cfg.hyde, cfg.diversity = True, 0.4
        why.append("analytical → dense + BM25, HyDE")
    else:
        why.append("factual → BM25 + dense")
    if a.complex:
        cfg.decompose = True
    if a.technical:
        cfg.weights["bm25"] = cfg.weights.get("bm25", 1.0) + 0.5
        why.append("technical terms → BM25 boosted")
    f_kw: dict = {}
    if a.time.hard:
        f_kw["date_from"] = a.time.start.isoformat() if a.time.start else None
        f_kw["date_to"] = a.time.end.isoformat() if a.time.end else None
        why.append(f"hard date filter {a.time.describe()}")
    if a.time.recency:
        cfg.freshness_weight = 0.35
        why.append("recency → freshness weighting")
        if len(a.entity_ids) == 1:
            f_kw["entity_any"] = frozenset(a.entity_ids)
            why.append("single entity + recency → entity filter")
    if a.wants_official:
        f_kw["source_types"] = frozenset({"official", "documentation"})
        why.append("asks for official sources → source filter")
    cfg.filters = Filters(**f_kw)
    cfg.describe = "; ".join(why)
    return cfg


def build_variants(a: QueryAnalysis, text: str, cfg: RetrievalConfig, llm: LLM | None, hyde_cache: dict[str, str]) -> list[Variant]:
    """original + rewrite + expansion (+ HyDE for dense), de-duplicated."""
    seen: set[str] = set()
    out: list[Variant] = []

    def add(t: str, w: float, kind: str) -> None:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(Variant(t, w, kind))

    add(text, 1.0, "original")
    if cfg.multi_query:
        sub_a = a  # entity-linked rewrite uses the whole analysis
        add(rewrite(sub_a) if text == a.original else text + " " + " ".join(a.entity_names), 0.7, "rewrite")
        ex = expand(text)
        if ex:
            add(text + " " + " ".join(ex), 0.5, "expansion")
    if cfg.hyde:
        h = hyde_cache.get(text)
        if h is None:
            h = (hyde_llm(a, llm) if llm else None) or hyde_template(a)
            hyde_cache[text] = h
        add(h, 0.6, "hyde")
    return out


def entity_type_names(a: QueryAnalysis, *types: EntityType) -> list[str]:
    return [n for i, n in zip(a.entity_ids, a.entity_names) if a.entity_types.get(i) in {str(t) for t in types}]
