"""HTML -> structured document, date parsing, content hashing, and section-level diffs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date

from bs4 import BeautifulSoup, Tag

from .domain import Block, ParsedDocument, Section
from .ingest import normalize_url

# ---------------------------------------------------------------- dates
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<![\w.-])(\d{4})-(\d{2})-(\d{2})(?!\d)"), "iso"),
    (re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", re.I), "mdy"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MON}\.?,?\s+((?:19|20)\d{{2}})\b", re.I), "dmy"),
    (re.compile(rf"\b{_MON}\.?,?\s+((?:19|20)\d{{2}})\b", re.I), "my"),
    (re.compile(r"\bQ([1-4])\s+((?:19|20)\d{2})\b"), "q"),
    (re.compile(r"(?<![\w.$-])((?:19|20)\d{2})(?![\w%.,-]?\d)(?!\s*(?:B|M|K|billion|million|tokens|parameters)\b)"), "y"),
]


@dataclass(slots=True)
class DateMention:
    start: date
    end: date
    precision: str  # day | month | quarter | year
    raw: str
    span: tuple[int, int]

    @property
    def iso(self) -> str:
        return self.start.isoformat()


def _last_day(y: int, m: int) -> date:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def extract_dates(text: str) -> list[DateMention]:
    """All date mentions in text, longest-pattern-wins on overlap, in text order."""
    found: list[DateMention] = []
    taken: list[tuple[int, int]] = []
    for pat, kind in DATE_PATTERNS:
        for m in pat.finditer(text):
            a, b = m.span()
            if any(a < tb and b > ta for ta, tb in taken):
                continue
            try:
                if kind == "iso":
                    d = date(int(m[1]), int(m[2]), int(m[3]))
                    dm = DateMention(d, d, "day", m[0], (a, b))
                elif kind == "mdy":
                    d = date(int(m[3]), MONTHS[m[1].lower().rstrip(".")], int(m[2]))
                    dm = DateMention(d, d, "day", m[0], (a, b))
                elif kind == "dmy":
                    d = date(int(m[3]), MONTHS[m[2].lower().rstrip(".")], int(m[1]))
                    dm = DateMention(d, d, "day", m[0], (a, b))
                elif kind == "my":
                    mo, y = MONTHS[m[1].lower().rstrip(".")], int(m[2])
                    dm = DateMention(date(y, mo, 1), _last_day(y, mo), "month", m[0], (a, b))
                elif kind == "q":
                    q, y = int(m[1]), int(m[2])
                    dm = DateMention(date(y, 3 * q - 2, 1), _last_day(y, 3 * q), "quarter", m[0], (a, b))
                else:
                    y = int(m[1])
                    dm = DateMention(date(y, 1, 1), date(y, 12, 31), "year", m[0], (a, b))
            except (ValueError, KeyError):
                continue
            found.append(dm)
            taken.append((a, b))
    return sorted(found, key=lambda d: d.span[0])


def parse_date(s: str | None) -> str | None:
    """Best-effort ISO date (YYYY-MM-DD) from a metadata string."""
    if not s:
        return None
    ds = extract_dates(s.strip())
    return ds[0].iso if ds else None


# ----------------------------------------------------------- HTML parsing
_NOISE_TAGS = ["script", "style", "noscript", "template", "iframe", "svg", "form", "nav", "footer", "button", "dialog"]
_NOISE_ATTR = re.compile(
    r"(^|[-_ ])(nav|navbar|menu|footer|sidebar|cookie|consent|banner|advert|ads?|promo|subscribe|newsletter|"
    r"share|social|breadcrumbs?|comments?|related|skip)([-_ ]|$)", re.I)
_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "table", "figcaption", "dd"]
_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _meta(soup: BeautifulSoup, *names: str) -> str | None:
    for n in names:
        t = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
        if t and t.get("content"):
            return str(t["content"]).strip()
    return None


def _jsonld(soup: BeautifulSoup) -> dict:
    for t in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            d = json.loads(t.string or "")
        except (ValueError, TypeError):
            continue
        for item in (d if isinstance(d, list) else [d]):
            if isinstance(item, dict) and ("datePublished" in item or "headline" in item):
                return item
    return {}


def _text_score(el: Tag) -> float:
    text = len(_clean(el.get_text(" ")))
    links = sum(len(_clean(a.get_text(" "))) for a in el.find_all("a"))
    return text - 2 * links


def _find_main(soup: BeautifulSoup) -> Tag:
    arts = soup.find_all("article")
    if arts:
        return max(arts, key=_text_score)
    for sel in ("main", '[role="main"]'):
        t = soup.select_one(sel)
        if t:
            return t
    divs = soup.find_all(["div", "section"])
    if divs:
        best = max(divs, key=_text_score)
        if _text_score(best) > 200:
            return best
    return soup.body or soup


def _strip_noise(soup: BeautifulSoup) -> None:
    for t in soup.find_all(_NOISE_TAGS):
        t.decompose()
    for t in soup.find_all(attrs={"role": re.compile("navigation|banner|contentinfo|complementary")}):
        t.decompose()
    for t in soup.find_all("aside"):
        t.decompose()
    for t in soup.find_all(True):
        if t.decomposed if hasattr(t, "decomposed") else False:
            continue
        if t.name in ("html", "body", "main", "article") or t.attrs is None:
            continue
        if t.find(["article", "main"]):
            continue
        sig = " ".join([*(t.get("class") or []), t.get("id") or ""])
        if sig and _NOISE_ATTR.search(sig):
            t.decompose()
    for t in soup.find_all("header"):
        if not t.find_parent(["article", "main"]):
            t.decompose()


def _own_text(el: Tag) -> str:
    parts = []
    for c in el.children:
        if isinstance(c, Tag):
            if c.name in ("ul", "ol"):
                continue
            parts.append(c.get_text(" "))
        else:
            parts.append(str(c))
    return _clean(" ".join(parts))


def _table_text(t: Tag) -> str:
    rows = []
    for tr in t.find_all("tr"):
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_html(html: str, url: str) -> ParsedDocument:
    soup = BeautifulSoup(html, "lxml")
    ld = _jsonld(soup)
    title = (_meta(soup, "og:title") or ld.get("headline") or (soup.title.string if soup.title and soup.title.string else "") or "")
    published = parse_date(_meta(soup, "article:published_time", "datePublished", "date", "dc.date")
                           or ld.get("datePublished") or (soup.find("time", attrs={"datetime": True}) or {}).get("datetime"))
    updated = parse_date(_meta(soup, "article:modified_time", "dateModified", "og:updated_time") or ld.get("dateModified"))
    canonical_tag = soup.find("link", rel="canonical")
    canonical = normalize_url(str(canonical_tag["href"]), url) if canonical_tag and canonical_tag.get("href") else None
    description = _meta(soup, "og:description", "description") or ""
    lang = (soup.html.get("lang") if soup.html else None) or "en"

    _strip_noise(soup)
    main = _find_main(soup)
    h1 = main.find("h1") or soup.find("h1")
    if not title and h1:
        title = _clean(h1.get_text(" "))
    title = _clean(re.split(r"\s+[|–—]\s+", _clean(title))[0]) if title else url

    links: list[str] = []
    for a in main.find_all("a", href=True):
        n = normalize_url(str(a["href"]), url)
        if n and n not in links:
            links.append(n)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    cur = Section(title, 1, (title,), [])
    sections.append(cur)
    nested_skip = ("li", "blockquote", "table", "pre")
    for el in main.find_all(_BLOCKS):
        if el.name in ("p", "figcaption", "dd") and el.find_parent(nested_skip):
            continue
        if el.name == "li" and el.find_parent("table"):
            continue
        if el.name in ("blockquote", "pre") and el.find_parent("pre"):
            continue
        if el.name[0] == "h" and len(el.name) == 2:
            text = _clean(el.get_text(" "))
            if not text:
                continue
            lvl = int(el.name[1])
            if lvl == 1 and text == title and not any(b for s in sections for b in s.blocks):
                continue  # the page title heading is the document title, not a section
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, text))
            cur = Section(text, lvl, tuple(t for _, t in stack), [])
            sections.append(cur)
            continue
        if el.name in ("li", "p") and (a := el.find("a")) is not None and _clean(a.get_text(" ")) == _clean(el.get_text(" ")):
            continue  # link-only item: navigation/listing, not prose (its href is still harvested as a link)
        if el.name == "table":
            txt, kind = _table_text(el), "table"
        elif el.name == "pre":
            txt, kind = el.get_text("\n").strip("\n"), "code"
        elif el.name == "li":
            txt, kind = _own_text(el), "li"
        elif el.name == "blockquote":
            txt, kind = _clean(el.get_text(" ")), "quote"
        else:
            txt, kind = _clean(el.get_text(" ")), "p"
        if txt:
            cur.blocks.append(Block(kind, txt))
    sections = [s for s in sections if s.blocks]
    return ParsedDocument(url, title, sections, published, updated, canonical, description, lang, links)


def to_markdown(doc: ParsedDocument) -> str:
    out = [f"# {doc.title}"]
    for s in doc.sections:
        if s.title != doc.title:
            out.append(f"\n{'#' * min(s.level + 1, 6)} {s.title}")
        for b in s.blocks:
            out.append({"li": f"- {b.text}", "code": f"```\n{b.text}\n```", "quote": f"> {b.text}"}.get(b.kind, b.text))
    return "\n".join(out)


# ------------------------------------------------- hashing & versioning
def normalize_text(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def content_hash(doc: ParsedDocument) -> str:
    """Hash of the *extracted* content, so ads/nonces/scripts never register as a change."""
    h = hashlib.sha256()
    h.update(normalize_text(doc.title).encode())
    for s in doc.sections:
        h.update(b"\x00" + normalize_text(" > ".join(s.path)).encode())
        for b in s.blocks:
            h.update(b"\x01" + b.kind.encode() + normalize_text(b.text).encode())
    return h.hexdigest()


def raw_hash(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8", "replace")).hexdigest()


@dataclass(slots=True)
class DocDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    title_changed: bool = False

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.modified or self.title_changed)

    def summary(self) -> str:
        if self.empty:
            return "no content change"
        parts = []
        for label, items in (("added", self.added), ("removed", self.removed), ("modified", self.modified)):
            if items:
                parts.append(f"{label} {len(items)} section(s): " + ", ".join(items[:4]) + ("…" if len(items) > 4 else ""))
        if self.title_changed:
            parts.insert(0, "title changed")
        return "; ".join(parts)


def diff_documents(old: ParsedDocument | None, new: ParsedDocument) -> DocDiff:
    if old is None:
        return DocDiff(added=[s.title for s in new.sections])
    key = lambda s: " > ".join(s.path)  # noqa: E731
    o = {key(s): normalize_text(s.text) for s in old.sections}
    n = {key(s): normalize_text(s.text) for s in new.sections}
    return DiffBuilder.build(o, n, new.title != old.title)


class DiffBuilder:
    @staticmethod
    def build(o: dict[str, str], n: dict[str, str], title_changed: bool) -> DocDiff:
        return DocDiff(
            added=[k.split(" > ")[-1] for k in n if k not in o],
            removed=[k.split(" > ")[-1] for k in o if k not in n],
            modified=[k.split(" > ")[-1] for k in n if k in o and o[k] != n[k]],
            title_changed=title_changed,
        )
