"""Knowledge processing: sentences, chunking, entity resolution/extraction, relations, temporal facts."""

from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
from dataclasses import dataclass

from .clock import Clock
from .config import Settings
from .domain import Chunk, Entity, EntityType, Mention, ParsedDocument, Relation
from .extract import extract_dates
from .ontology import (
    CATEGORY_ENTITIES,
    CREATABLE,
    FAMILY_ENTITY,
    MODEL_FAMILIES,
    RELATIONS,
    SEED_ENTITIES,
)
from .store import Repository

# ------------------------------------------------------------ sentences
_ABBREV = {"inc", "corp", "ltd", "co", "vs", "dr", "mr", "ms", "st", "no", "e.g", "i.e", "etc", "u.s", "approx", "fig"}
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """(start, end, sentence) spans. Guards common abbreviations; decimals never split (no whitespace)."""
    out, start = [], 0
    for m in _SPLIT.finditer(text):
        prev = text[start:m.start()].rsplit(None, 1)[-1].rstrip(".").lower() if text[start:m.start()].strip() else ""
        if prev in _ABBREV:
            continue
        out.append((start, m.start(), text[start:m.start()]))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text), text[start:]))
    return out


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


# ------------------------------------------------------------- chunking
def chunk_id_for(document_id: str, section_path: str, text: str, occurrence: int = 0) -> str:
    """Content-addressed: an unchanged paragraph keeps its id across versions => no re-embedding."""
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(f"{document_id}|{section_path}|{norm}|{occurrence}".encode()).hexdigest()[:16]


def chunk_document(doc: ParsedDocument, document_id: str, settings: Settings, meta: dict) -> list[Chunk]:
    """Structure-aware chunking: never crosses a section boundary; code/tables stay atomic;
    long paragraphs split at sentence boundaries with a one-sentence overlap."""
    target, hard_max, min_tok = settings.chunk_target_tokens, settings.chunk_max_tokens, settings.chunk_min_tokens
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    def emit(section_path: str, section_title: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        base = chunk_id_for(document_id, section_path, text)
        occ = seen.get(base, 0)
        seen[base] = occ + 1
        cid = base if occ == 0 else chunk_id_for(document_id, section_path, text, occ)
        chunks.append(Chunk(cid, document_id, text, section_title, len(chunks), count_tokens(text), **meta))

    for sec in doc.sections:
        path = " > ".join(sec.path)
        buf: list[str] = []
        size = 0

        def flush(buf=buf, path=path, sec=sec) -> None:
            if buf:
                emit(path, sec.title, "\n".join(buf))
                buf.clear()

        for b in sec.blocks:
            if b.kind in ("code", "table"):
                flush()
                size = 0
                if count_tokens(b.text) <= hard_max:
                    emit(path, sec.title, b.text)
                else:
                    lines, part, n = b.text.split("\n"), [], 0
                    for ln in lines:
                        if n + count_tokens(ln) > target and part:
                            emit(path, sec.title, "\n".join(part))
                            part, n = [], 0
                        part.append(ln)
                        n += count_tokens(ln)
                    emit(path, sec.title, "\n".join(part))
                continue
            pieces = [b.text]
            if count_tokens(b.text) > hard_max:
                pieces, cur, n = [], [], 0
                for _, _, s in split_sentences(b.text):
                    st = count_tokens(s)
                    if n + st > target and cur:
                        pieces.append(" ".join(cur))
                        cur, n = cur[-1:], count_tokens(cur[-1])  # overlap
                    cur.append(s)
                    n += st
                if cur:
                    pieces.append(" ".join(cur))
            for piece in pieces:
                pt = count_tokens(piece)
                if size and size + pt > target:
                    flush()
                    size = 0
                buf.append(("- " if b.kind == "li" else "") + piece)
                size += pt
        flush()
    # merge runt chunks into their predecessor within the same section
    merged: list[Chunk] = []
    for c in chunks:
        if merged and c.token_count < min_tok and merged[-1].section_title == c.section_title \
                and merged[-1].token_count + c.token_count <= hard_max:
            p = merged[-1]
            text = p.text + "\n" + c.text
            p.chunk_id = chunk_id_for(document_id, c.section_title, text)
            p.text, p.token_count = text, count_tokens(text)
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c.ordinal = i
    return merged


# --------------------------------------------------- entity resolution
_CORP = {"inc", "corp", "corporation", "ltd", "llc", "plc", "co", "company", "limited", "pbc"}


def alias_key(text: str) -> str:
    """Normal form under which surface variants collapse:
    'OpenAI, Inc.' / "OpenAI's" -> 'openai'; 'GPT-4' / 'GPT 4' / 'GPT4' -> 'gpt4'; 'Llama 3.1' -> 'llama3p1'."""
    s = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    words = re.findall(r"[a-z0-9.+]+", s)
    while len(words) > 1 and words[-1].strip(".") in _CORP:
        words.pop()
    if words and words[0] == "the":
        words = words[1:]
    s = "".join(words)
    s = re.sub(r"(?<=\d)\.(?=\d)", "p", s)
    return re.sub(r"[^a-z0-9+]", "", s)


def entity_id_for(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


_LENIENT_BLOCK = {"math", "rl", "mla", "swa", "gqa", "fair"}
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-+'’][A-Za-z0-9]+)*\+*")


class EntityResolver:
    def __init__(self, repo: Repository, clock: Clock) -> None:
        self.repo, self.clock = repo, clock
        self._gaz: dict[str, tuple[str, str, bool]] = {}  # key -> (entity_id, type, lowercase_ok)
        self._types: dict[str, str] = {}
        self._by_key_type: dict[str, list[str]] = {}
        self.reload()

    # -- seeding
    def seed(self) -> None:
        ts = self.clock.now_iso()
        for name, etype, desc, aliases in SEED_ENTITIES:
            eid = entity_id_for(name)
            self.repo.upsert_entity(Entity(eid, name, etype, desc, False), ts)
            for a in [name, *aliases]:
                self.repo.add_alias(eid, alias_key(a), a)
        self.reload()

    def reload(self) -> None:
        self._gaz.clear()
        self._types.clear()
        for r in self.repo.alias_rows():
            txt = r["alias_text"]
            multiword = len(txt.split()) > 1 or "-" in txt
            acronym = len(txt) <= 5 and (txt.isupper() or any(c.isupper() for c in txt[1:]))  # RAG, MoE, LoRA, GQA
            lower_ok = multiword or (r["type"] == EntityType.TECHNOLOGY and not acronym)
            self._gaz[r["alias_key"]] = (r["entity_id"], r["type"], lower_ok)
            self._types[r["entity_id"]] = r["type"]

    def type_of(self, entity_id: str) -> str | None:
        return self._types.get(entity_id)

    def lookup(self, surface: str) -> str | None:
        hit = self._gaz.get(alias_key(surface))
        return hit[0] if hit else None

    def resolve(self, surface: str, type_hint: str | None = None, create: bool = True) -> str | None:
        """exact alias -> fuzzy (same type, ratio>=0.93) -> create provisional entity."""
        key = alias_key(surface)
        if not key:
            return None
        if key in self._gaz:
            return self._gaz[key][0]
        if len(key) >= 5:
            best, score = None, 0.0
            for k, (eid, typ, _) in self._gaz.items():
                if type_hint and typ != type_hint:
                    continue
                r = difflib.SequenceMatcher(None, key, k).ratio()
                if r > score:
                    best, score = eid, r
            # digits must agree exactly: 'gpt4' must never fuzzy-match 'gpt5'
            if best and score >= 0.93:
                bk = next(k for k, v in self._gaz.items() if v[0] == best and difflib.SequenceMatcher(None, key, k).ratio() == score)
                if re.findall(r"\d+", bk) == re.findall(r"\d+", key):
                    self.add_alias(best, surface)
                    return best
        if not create or not type_hint or type_hint not in CREATABLE:
            return None
        eid = entity_id_for(surface)
        self.repo.upsert_entity(Entity(eid, surface.strip(), type_hint, "", True), self.clock.now_iso())
        self.repo.add_alias(eid, key, surface)
        self._gaz[key] = (eid, type_hint, True)
        self._types[eid] = type_hint
        return eid

    def add_alias(self, entity_id: str, surface: str) -> None:
        self.repo.add_alias(entity_id, alias_key(surface), surface)
        self._gaz.setdefault(alias_key(surface), (entity_id, self._types[entity_id], True))

    # -- text -> mentions
    def extract(self, text: str, discover: bool = True, lenient: bool = False) -> list[Mention]:
        """lenient=True ignores capitalisation (user queries are often lower-case)."""
        toks = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text)]
        mentions: list[Mention] = []
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(6, len(toks) - i), 0, -1):
                a, b = toks[i][0], toks[i + n - 1][1]
                surface = text[a:b]
                if n > 1 and toks[i][2].lower() in ("the", "a", "an"):
                    continue  # "the math" must not match the alias "MATH" through alias_key's article stripping
                # tokens must be separated by plain whitespace for multi-word matches
                if n > 1 and re.search(r"[^\s\-]", "".join(text[toks[j][1]:toks[j + 1][0]] for j in range(i, i + n - 1))):
                    continue
                g = self._gaz.get(alias_key(surface))
                if g and (self._surface_ok(surface, n, g[2]) or (lenient and surface.lower() not in _LENIENT_BLOCK)):
                    hit = (n, g[0], g[1], surface, a, b)
                    break
            if hit:
                n, eid, etype, surface, a, b = hit
                mentions.append(Mention(eid, surface, a, b, etype))
                i += n
                continue
            i += 1
        if discover:
            mentions = self._discover(text, mentions)
        return sorted(mentions, key=lambda m: m.start)

    @staticmethod
    def _surface_ok(surface: str, ntok: int, lower_ok: bool) -> bool:
        if lower_ok or ntok > 1:
            return True
        return surface[:1].isupper() or any(c.isdigit() for c in surface)

    _MODEL_RE = re.compile(
        r"\b(" + "|".join(MODEL_FAMILIES) + r")[\s-]?v?(\d+(?:\.\d+)?)([A-Za-z]{0,3})?"
        r"(?:[\s-](Scout|Maverick|Behemoth|Mini|Turbo|Pro|Ultra|Nano|Flash|Opus|Sonnet|Haiku|Instruct|Chat|Coder|R1|V3|Preview))?\b")

    def _discover(self, text: str, mentions: list[Mention]) -> list[Mention]:
        """Unseen model versions (e.g. 'Llama 3.2') become provisional MODEL entities linked to their family."""
        out = list(mentions)
        for m in self._MODEL_RE.finditer(text):
            a, b = m.span()
            overlap = [x for x in out if a < x.end and b > x.start]
            if any(not (x.start >= a and x.end <= b and x.type == EntityType.MODEL) for x in overlap):
                continue  # clashes with a different entity
            surface = m.group(0)
            if alias_key(surface) in self._gaz:
                continue
            eid = self.resolve(surface, EntityType.MODEL)
            if eid and FAMILY_ENTITY.get(m.group(1)):
                out = [x for x in out if x not in overlap]  # "Llama 3.2" supersedes the bare family mention "Llama"
                out.append(Mention(eid, surface, a, b, EntityType.MODEL))
        return out


# ------------------------------------------------------------- relations
_ADV = r"(?:(?:also|first|now|today|officially|later|then|recently|has|have|had|is|was|were|been|already|formally|jointly|continues to|continued to)\s+)*"
_FILL = (r"(?:(?:the|a|an|its|their|his|her|new|first|latest|next[- ]generation|flagship|family|series|of|open|open-weight|"
         r"open-source|multimodal|large|language|model|models|reasoning|generation|version|called|named|dubbed|"
         r"fully|brand-new|in-house|custom|own|and|to|more|most|powerful|AI|ai)\s*,?\s*){0,8}")
_ANY = r"(?:[\w\-/%.$,]+\s+){0,4}"

_ACTIVE: list[tuple[str, str, float]] = [
    ("RELEASED", r"(?:released|releases|releasing|launched|launches|introduced|introduces|unveiled|unveils|announced|"
                 r"announces|open[- ]sourced|published|debuted|shipped|rolled out|presented|presents)\s+" + _FILL, 0.85),
    ("DEVELOPS", r"(?:develops|developed|developing|builds|built|building|designs|designed|created|creates|makes|made|"
                 r"produces|trains|trained|developed and released)\s+" + _FILL, 0.8),
    ("ACQUIRED", r"(?:acquired|acquires|bought|purchased|agreed to acquire|to acquire|will acquire|has acquired)\s+" + _FILL, 0.9),
    ("PARTNERS_WITH", r"(?:partnered|partners|partnering|collaborated|collaborates|collaborating|teamed up|teams up|allied|"
                      r"joined forces|working|works|worked|cooperates)\s+(?:closely\s+)?with\s+|(?:in|a)?\s*(?:partnership|collaboration|"
                      r"alliance)\s+with\s+|(?:and)\s+(?:announced\s+)?(?:a\s+)?(?:strategic\s+)?(?:partnership|collaboration)\s*,?\s*", 0.85),
    ("INVESTED_IN", r"(?:invested|invests|investing|committed)\s+(?:\$?[\d.,]+\s*(?:billion|million|bn|m)\s+)?(?:in|into)\s+|"
                    r"(?:backed|funded|backs|funds)\s+", 0.85),
    ("BASED_ON", r"(?:is|are|was|were)?\s*(?:built|based|builds|relies|relying)\s+(?:on|upon)\s+" + _ANY, 0.8),
    ("USES", r"(?:uses|use|used|using|utilizes|utilises|employs|adopts|leverages|incorporates|features|featuring|implements|"
             r"introduces|integrates|combines|supports|adopt|rely|relies on)\s+" + _ANY, 0.8),
    ("TRAINED_ON", r"(?:was|were|is|are|been)?\s*(?:pre-?trained|trained)\s+(?:\w+\s+){0,3}(?:on|using|with)\s+" + _ANY, 0.8),
    ("EVALUATED_ON", r"(?:evaluated|tested|benchmarked|scores?|scored|achieves?|achieved|reaches|reached|reports?)\s+" + _ANY + r"(?:on|in|against)\s+" + _ANY, 0.75),
    ("OUTPERFORMS", r"(?:outperforms?|outperformed|surpass(?:es|ed)|beats?|exceeds?|exceeded|tops?)\s+" + _FILL, 0.8),
    ("INTRODUCED", r"(?:introduced|introduces|proposed|proposes|presented|presents)\s+" + _FILL, 0.85),
    ("VARIANT_OF", r"(?:is|are)\s+(?:a|an|the)\s+(?:\w+\s+){0,2}(?:variant|version|member|release)\s+of\s+", 0.8),
]
_PASSIVE: list[tuple[str, str, float]] = [
    ("RELEASED", r"(?:was|were|is|has been|have been|had been)\s+(?:also\s+|first\s+|recently\s+|officially\s+)*"
                 r"(?:released|launched|introduced|unveiled|announced|published|open[- ]sourced|presented)\s+by\s+", 0.85),
    ("DEVELOPS", r"(?:was|were|is|are|has been|have been)\s+(?:also\s+|primarily\s+)*(?:developed|built|created|designed|"
                 r"trained|made|produced|engineered)\s+by\s+", 0.85),
    ("ACQUIRED", r"(?:was|were|is|has been)\s+(?:also\s+)?(?:acquired|bought|purchased)\s+by\s+", 0.9),
    ("INVESTED_IN", r"(?:was|were|is|has been)\s+(?:also\s+)?(?:backed|funded)\s+by\s+", 0.85),
    ("USES", r"(?:is|are|was|were)\s+(?:also\s+)?(?:used|employed|adopted|utilized|implemented)\s+(?:in|by)\s+", 0.75),
]
_PREP: list[tuple[str, str, float]] = [("DEVELOPS", r"(?:from|by)\s*$", 0.7)]  # "Llama 3 from Meta"
_POSS = re.compile(r"^(?:'s|’s)?\s*$")
_CONJ = re.compile(r"^\s*(?:,|,?\s*and|,?\s*as well as|,?\s*along with)\s*$", re.I)
_ISA = re.compile(r"^\s*(?:,|\()?\s*(?:is|are|was|were)\s+(?:an?|the|one of the)\b", re.I)


@dataclass(slots=True)
class ExtractedRelation:
    src_id: str
    rel: str
    dst_id: str
    sentence: str
    confidence: float
    valid_from: str | None


class RelationExtractor:
    def __init__(self, resolver: EntityResolver) -> None:
        self.res = resolver
        self._active = [(r, re.compile(r"^\s*" + _ADV + p + r"\s*$", re.I), c) for r, p, c in _ACTIVE]
        self._passive = [(r, re.compile(r"^\s*" + p + r"\s*$", re.I), c) for r, p, c in _PASSIVE]
        self._prep = [(r, re.compile(p, re.I), c) for r, p, c in _PREP]

    def _ok(self, rel: str, a: Mention, b: Mention) -> bool:
        if a.entity_id == b.entity_id:
            return False
        dom, rng = RELATIONS[rel]
        return a.type in {str(x) for x in dom} and b.type in {str(x) for x in rng}

    def extract_sentence(self, sent: str, mentions: list[Mention], doc_date: str | None) -> list[ExtractedRelation]:
        out: dict[tuple[str, str, str], ExtractedRelation] = {}
        dates = extract_dates(sent)
        vfrom = dates[0].iso if dates else doc_date

        def add(src: Mention, rel: str, dst: Mention, conf: float) -> None:
            k = (src.entity_id, rel, dst.entity_id)
            if k not in out or out[k].confidence < conf:
                out[k] = ExtractedRelation(src.entity_id, rel, dst.entity_id, sent.strip(), round(conf, 3), vfrom)

        last: tuple[Mention, str, float] | None = None  # for "X released A and B"
        for i in range(len(mentions) - 1):
            a, b = mentions[i], mentions[i + 1]
            between = sent[a.end:b.start]
            if len(between) > 160:
                last = None
                continue
            matched = False
            for rel, rx, conf in self._active:
                if rx.match(between) and self._ok(rel, a, b):
                    add(a, rel, b, conf)
                    last, matched = (a, rel, conf), True
                    break
            if not matched:
                for rel, rx, conf in self._passive:
                    if rx.match(between) and self._ok(rel, b, a):
                        add(b, rel, a, conf)
                        matched = True
                        break
            if not matched and last is not None and _CONJ.match(between) and self._ok(last[1], last[0], b):
                add(last[0], last[1], b, last[2] * 0.95)  # conjunction: "Meta released Llama 3 and Llama 3.1"
                matched = True
            if not matched and _POSS.match(between) and self._ok("DEVELOPS", a, b) and a.surface.endswith(("'s", "’s")):
                add(a, "DEVELOPS", b, 0.55)  # "Meta's Llama 3"
                matched = True
            if not matched:
                for rel, rx, conf in self._prep:
                    if rx.search(between) and self._ok(rel, b, a):
                        add(b, rel, a, conf)
                        matched = True
                        break
            if not matched:
                last = None if not _CONJ.match(between) else last
        # IS_A: "<A> is a ... <category>" regardless of intervening mentions
        for i, a in enumerate(mentions):
            if not _ISA.match(sent[a.end:a.end + 40]):
                continue
            for b in mentions[i + 1:]:  # only the nearest category counts: "an AI accelerator designed to train LLMs"
                ent = self.res.repo.entity(b.entity_id)
                if ent and ent.canonical_name in CATEGORY_ENTITIES and self._ok("IS_A", a, b):
                    if len(sent[a.end:b.start]) < 100:
                        add(a, "IS_A", b, 0.8)
                    break
        # VARIANT_OF: provisional model named after a known family
        for m in mentions:
            ent = self.res.repo.entity(m.entity_id)
            if ent and ent.provisional and ent.type == EntityType.MODEL:
                fam = next((FAMILY_ENTITY[f] for f in FAMILY_ENTITY if ent.canonical_name.startswith(f)), None)
                fid = entity_id_for(fam) if fam else None
                if fid and fid != m.entity_id and self.res.type_of(fid):
                    out[(m.entity_id, "VARIANT_OF", fid)] = ExtractedRelation(m.entity_id, "VARIANT_OF", fid, sent.strip(), 0.9, vfrom)
        return list(out.values())


def relation_id(src: str, rel: str, dst: str, document_id: str) -> str:
    return hashlib.sha1(f"{src}|{rel}|{dst}|{document_id}".encode()).hexdigest()[:16]


def build_relation(x: ExtractedRelation, document_id: str, chunk_id: str | None, observed_at: str, version: int) -> Relation:
    return Relation(relation_id(x.src_id, x.rel, x.dst_id, document_id), x.src_id, x.rel, x.dst_id, document_id,
                    chunk_id, x.sentence, x.confidence, observed_at, x.valid_from, None, True, version, version)

