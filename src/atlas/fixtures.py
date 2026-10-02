"""Offline fixture web: a small, self-contained AI/technology corpus served through httpx.MockTransport.

Pages are concise paraphrases of public announcements, hosted on reserved `.example` domains so they can never
be mistaken for the real sites. Round 2 adds, edits and removes pages to exercise incremental updates.
"""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass, field

import httpx

from .domain import Source, SourceType

OFF, RES, DOC, BLOG, NEWS, COMM = (SourceType.OFFICIAL, SourceType.RESEARCH, SourceType.DOCUMENTATION,
                                   SourceType.TECHNICAL_BLOG, SourceType.NEWS, SourceType.COMMUNITY)

# host -> (source name, source type, priority)
HOSTS: dict[str, tuple[str, SourceType, int]] = {
    "openai.example": ("OpenAI", OFF, 9),
    "anthropic.example": ("Anthropic", OFF, 9),
    "meta-ai.example": ("Meta AI", OFF, 9),
    "deepmind.example": ("Google DeepMind", OFF, 9),
    "mistral.example": ("Mistral AI", OFF, 8),
    "deepseek.example": ("DeepSeek", OFF, 8),
    "qwen.example": ("Alibaba Qwen", OFF, 7),
    "nvidia.example": ("NVIDIA", OFF, 9),
    "cloud-google.example": ("Google Cloud", OFF, 8),
    "aws.example": ("Amazon Web Services", OFF, 8),
    "azure.example": ("Microsoft Azure", OFF, 8),
    "amd.example": ("AMD", OFF, 7),
    "arxiv.example": ("arXiv", RES, 8),
    "hf.example": ("Hugging Face Blog", BLOG, 6),
    "hf-docs.example": ("Hugging Face Docs", DOC, 7),
    "techwire.example": ("TechWire", NEWS, 5),
    "forum.example": ("AI Forum", COMM, 3),
}


@dataclass
class Page:
    key: str
    host: str
    path: str
    title: str
    published: str
    sections: list[tuple[str, list]]
    updated: str | None = None
    first_round: int = 1
    jsonld: bool = False
    bare_div: bool = False  # no <article>/<main>: forces the text-density fallback

    @property
    def url(self) -> str:
        return f"https://{self.host}{self.path}"


def _ul(*items: str) -> tuple[str, list[str]]:
    return ("ul", list(items))


PAGES: list[Page] = [
    Page("gpt-4", "openai.example", "/news/gpt-4", "Introducing GPT-4", "2023-03-14", [
        ("Overview", [
            "OpenAI released GPT-4 on March 14, 2023. GPT-4 is a large multimodal model that accepts image and text inputs and produces text outputs.",
            "GPT-4 is based on the Transformer architecture and was pre-trained to predict the next token in a document."]),
        ("Capabilities", [
            "GPT-4 exhibits human-level performance on various professional benchmarks, including a simulated bar exam where it scores around the top 10% of test takers.",
            "OpenAI evaluated GPT-4 on MMLU and reports strong results across many languages."]),
        ("Availability", ["GPT-4 is available through ChatGPT Plus and the OpenAI API. Microsoft confirmed that its Bing chat experience runs on GPT-4."]),
    ], jsonld=True),
    Page("gpt-4o", "openai.example", "/news/gpt-4o", "Hello GPT-4o", "2024-05-13", [
        ("Overview", [
            "OpenAI introduced GPT-4o on May 13, 2024. GPT-4o is a flagship omni model that can reason across audio, vision and text in real time.",
            "GPT-4o is faster and cheaper than GPT-4 Turbo in the API, and it improves on GPT-4 for vision and non-English languages."]),
        ("Availability", ["GPT-4o is rolling out in ChatGPT, and developers can use GPT-4o in the OpenAI API."]),
    ]),
    Page("openai-microsoft", "openai.example", "/news/microsoft-partnership", "OpenAI and Microsoft extend partnership", "2023-01-23", [
        ("The partnership", [
            "Microsoft announced a multiyear, multibillion dollar investment in OpenAI in January 2023. Microsoft invested in OpenAI to accelerate AI breakthroughs and share them broadly.",
            "OpenAI partners with Microsoft to run its models on Azure. Azure is the exclusive cloud provider for ChatGPT and the OpenAI API."]),
    ]),
    Page("claude-3", "anthropic.example", "/news/claude-3-family", "Introducing the Claude 3 family", "2024-03-04", [
        ("Overview", [
            "Anthropic released the Claude 3 family on March 4, 2024. The family includes three models in ascending order of capability: Claude 3 Haiku, Claude 3 Sonnet and Claude 3 Opus.",
            "Claude 3 Opus outperforms GPT-4 on common evaluation benchmarks such as MMLU, HumanEval and GSM8K."]),
        ("Multimodal", ["The Claude 3 models are multimodal and can process image inputs such as photos, charts and documents."]),
    ]),
    Page("mcp", "anthropic.example", "/news/model-context-protocol", "Introducing the Model Context Protocol", "2024-11-25", [
        ("Overview", [
            "Anthropic open-sourced the Model Context Protocol on November 25, 2024. MCP is an open standard that connects AI assistants to the systems where data lives, including content repositories, business tools and development environments.",
            "Developers can build MCP servers that expose their data, and AI applications act as MCP clients. Anthropic released MCP servers for Google Drive, Slack and GitHub."]),
    ]),
    Page("claude-code", "anthropic.example", "/product/claude-code", "Claude Code: an agentic coding tool", "2025-02-24", [
        ("Overview", [
            "Anthropic released Claude Code as a limited research preview in February 2025. Claude Code is an agentic coding tool that works directly in the terminal.",
            "Claude Code supports the Model Context Protocol, so developers can connect external tools and data sources."]),
    ]),
    Page("llama-2", "meta-ai.example", "/blog/llama-2", "Llama 2: open foundation and chat models", "2023-07-18", [
        ("Overview", [
            "Meta released Llama 2 on July 18, 2023. Llama 2 is free for research and commercial use, and the models range from 7 billion to 70 billion parameters.",
            "Llama 2 was trained on 2 trillion tokens and doubles the context length of Llama to 4096 tokens."]),
        ("Partners", ["Meta partnered with Microsoft to make Llama 2 available through Azure. Microsoft is the preferred partner for Llama 2."]),
    ]),
    Page("llama-3", "meta-ai.example", "/blog/llama-3", "Introducing Meta Llama 3", "2024-04-18", [
        ("Overview", [
            "Meta introduced Meta Llama 3 on April 18, 2024. Llama 3 comes in 8B and 70B parameter sizes and was trained on over 15 trillion tokens.",
            "Llama 3 uses Grouped-Query Attention to improve inference efficiency. Meta describes Llama 3 as the most capable openly available large language model to date."]),
    ]),
    Page("llama-3-1", "meta-ai.example", "/blog/llama-3-1", "Llama 3.1: open source AI goes frontier", "2024-07-23", [
        ("Overview", [
            "Meta released Llama 3.1 on July 23, 2024, including a 405B parameter model. Llama 3.1 405B is the first openly available model that rivals the top closed models.",
            "Llama 3.1 extends the context length to 128K tokens."]),
        ("Ecosystem", ["Meta open-sourced Llama 3.1 under a permissive license. Meta partnered with NVIDIA, Microsoft and Amazon Web Services to support developers building on Llama 3.1."]),
    ]),
    Page("llama-4", "meta-ai.example", "/blog/llama-4", "The Llama 4 herd", "2025-04-05", [
        ("Overview", [
            "Meta released Llama 4 Scout and Llama 4 Maverick on April 5, 2025. Llama 4 uses a Mixture of Experts architecture and both models are natively multimodal.",
            "Meta also previewed Llama 4 Behemoth, a teacher model that was still in training."]),
        ("Models", [("table", [["Model", "Active parameters", "Experts", "Context window"],
                               ["Llama 4 Scout", "17B", "16", "10M tokens"],
                               ["Llama 4 Maverick", "17B", "128", "1M tokens"]]),
                    "Llama 4 Scout fits on a single NVIDIA H100 GPU with quantization and offers an industry-leading context window of 10 million tokens."]),
    ], first_round=2),
    Page("llama-hub", "meta-ai.example", "/models/llama", "The Llama model family", "2024-04-18", [
        ("Overview", ["Meta develops the Llama family of open-weight models. Llama 2 and Llama 3 are available for download."]),
        ("Current models", [_ul("Llama 2: previous generation", "Llama 3: current generation")]),
    ], bare_div=True),
    Page("mistral-7b", "mistral.example", "/news/announcing-mistral-7b", "Mistral 7B", "2023-09-27", [
        ("Overview", [
            "Mistral AI released Mistral 7B on September 27, 2023. Mistral 7B is a 7.3 billion parameter model released under the Apache 2.0 license.",
            "Mistral 7B uses Grouped-Query Attention for faster inference and Sliding Window Attention to handle longer sequences at lower cost.",
            "Mistral 7B outperforms Llama 2 on all evaluated benchmarks."]),
    ]),
    Page("mixtral", "mistral.example", "/news/mixtral-of-experts", "Mixtral of experts", "2023-12-11", [
        ("Overview", [
            "Mistral AI released Mixtral 8x7B on December 11, 2023. Mixtral is a sparse Mixture of Experts model with open weights under the Apache 2.0 license.",
            "Mixtral uses a Mixture of Experts architecture in which a router network selects two experts per token.",
            "Mixtral outperforms Llama 2 on most benchmarks with 6x faster inference."]),
    ]),
    Page("mistral-about", "mistral.example", "/company", "About Mistral AI", "2023-09-27", [
        ("Company", ["Mistral AI is a Paris-based company building open-weight models. Mistral AI develops Mistral 7B and Mixtral. The company was founded in 2023 and releases many of its models under permissive licenses."]),
    ]),
    Page("gemini", "deepmind.example", "/blog/introducing-gemini", "Introducing Gemini", "2023-12-06", [
        ("Overview", [
            "Google introduced Gemini on December 6, 2023. Gemini was built from the ground up to be natively multimodal and comes in three sizes: Gemini Ultra, Gemini Pro and Gemini Nano.",
            "Gemini Ultra achieves 90.0% on MMLU, the first model to outperform human experts on that benchmark."]),
    ]),
    Page("gemma", "deepmind.example", "/blog/gemma", "Gemma: open models based on Gemini research", "2024-02-21", [
        ("Overview", [
            "Google released Gemma on February 21, 2024. Gemma is a family of lightweight open models available in 2B and 7B sizes.",
            "Gemma is based on Gemini research and technology, and Google released model weights for both sizes."]),
    ]),
    Page("tpu-v5p", "cloud-google.example", "/blog/tpu-v5p", "Introducing Cloud TPU v5p", "2023-12-06", [
        ("Overview", [
            "Google announced Cloud TPU v5p on December 6, 2023. TPU v5p is an AI accelerator designed to train large language models.",
            "Google builds TPU hardware for training and serving models, including Gemini."]),
    ]),
    Page("blackwell", "nvidia.example", "/news/blackwell-platform", "NVIDIA Blackwell platform arrives", "2024-03-18", [
        ("Overview", [
            "NVIDIA announced the Blackwell GPU architecture on March 18, 2024. Blackwell is an AI accelerator platform built for trillion-parameter models.",
            "NVIDIA is working with Amazon Web Services, Google, Microsoft and Meta on Blackwell deployments."]),
    ]),
    Page("hopper", "nvidia.example", "/news/hopper-h100", "NVIDIA announces Hopper architecture", "2022-03-22", [
        ("Overview", [
            "NVIDIA announced the Hopper architecture and the H100 GPU in March 2022. The H100 is an AI accelerator built to speed up Transformer models.",
        ]),
    ]),
    Page("nvidia-google", "nvidia.example", "/news/google-cloud-partnership", "NVIDIA and Google expand partnership", "2024-04-09", [
        ("Overview", ["NVIDIA partners with Google to bring Blackwell to Google Cloud. NVIDIA and Google also optimize Gemma for NVIDIA GPUs."]),
    ]),
    Page("trainium2", "aws.example", "/news/trainium2", "AWS announces Trainium2", "2023-11-28", [
        ("Overview", [
            "Amazon Web Services announced Trainium2 on November 28, 2023. Trainium2 is an AI accelerator designed to train foundation models, offering up to 4x faster training than the first-generation Trainium.",
            "AWS collaborates with NVIDIA to offer NVIDIA GPU instances alongside its own silicon."]),
    ]),
    Page("maia", "azure.example", "/news/maia-100", "Microsoft unveils Maia 100", "2023-11-15", [
        ("Overview", [
            "Microsoft announced Maia 100 on November 15, 2023. Maia 100 is an AI accelerator that Microsoft designed for large language models on Azure.",
            "Microsoft also partners with NVIDIA and AMD for GPU capacity on Azure."]),
    ]),
    Page("mi300x", "amd.example", "/news/instinct-mi300x", "AMD launches Instinct MI300X", "2023-12-06", [
        ("Overview", [
            "AMD launched the Instinct MI300X on December 6, 2023. The Instinct MI300X is an AI accelerator with 192 GB of HBM3 memory.",
            "AMD works with Microsoft and Meta on MI300X deployments."]),
    ]),
    Page("deepseek-v3", "deepseek.example", "/news/deepseek-v3", "DeepSeek-V3 technical report", "2024-12-26", [
        ("Overview", [
            "DeepSeek released DeepSeek-V3 on December 26, 2024. DeepSeek-V3 is a Mixture of Experts model with 671 billion total parameters, of which 37 billion are activated per token.",
            "DeepSeek-V3 uses Multi-head Latent Attention for efficient inference. DeepSeek-V3 outperforms Llama 3.1 on several benchmarks including MMLU and MATH."]),
    ]),
    Page("deepseek-r1", "deepseek.example", "/news/deepseek-r1", "DeepSeek-R1: reasoning via reinforcement learning", "2025-01-20", [
        ("Overview", [
            "DeepSeek released DeepSeek-R1 on January 20, 2025. DeepSeek-R1 was trained with Reinforcement Learning to improve reasoning.",
            "DeepSeek-R1 is based on DeepSeek-V3. DeepSeek open-sourced DeepSeek-R1 under the MIT license."]),
    ], first_round=2),
    Page("qwen25", "qwen.example", "/blog/qwen2-5", "Qwen2.5: a party of foundation models", "2024-09-19", [
        ("Overview", ["Alibaba Cloud released Qwen2.5 on September 19, 2024. Qwen2.5 is a family of open-weight models ranging from 0.5B to 72B parameters."]),
    ]),
    Page("attention", "arxiv.example", "/abs/1706.03762", "Attention Is All You Need", "2017-06-12", [
        ("Abstract", [
            "The paper Attention Is All You Need introduced the Transformer in June 2017. The paper was written by researchers at Google.",
            "The Transformer relies on attention mechanisms alone, dispensing with recurrence and convolutions. The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German translation task."]),
    ]),
    Page("lora", "arxiv.example", "/abs/2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", "2021-06-17", [
        ("Abstract", [
            "The LoRA paper introduced Low-Rank Adaptation, a method that freezes pretrained model weights and injects trainable low-rank matrices into Transformer layers.",
            "LoRA reduces the number of trainable parameters by 10,000 times compared with full Fine-tuning of GPT-3. The paper was written by researchers at Microsoft."]),
    ]),
    Page("rag-paper", "arxiv.example", "/abs/2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "2020-05-22", [
        ("Abstract", [
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks introduced Retrieval-Augmented Generation (RAG), which combines a pretrained sequence-to-sequence model with a dense vector index of Wikipedia.",
            "The paper was written by researchers at Facebook AI Research, University College London and New York University. RAG models set state-of-the-art results on open-domain question answering."]),
    ]),
    Page("moe-explained", "hf.example", "/blog/moe", "Mixture of Experts Explained", "2023-12-11", [
        ("What is a Mixture of Experts?", [
            "A Mixture of Experts model replaces dense feed-forward layers in a Transformer with sparse expert layers. A gating network routes each token to a small number of experts, so only a fraction of the parameters is active per token.",
            "Mixtral and DeepSeek-V3 are examples of Mixture of Experts models."]),
        ("Trade-offs", ["Mixture of Experts models pretrain faster than dense models and offer faster inference for their size, but they need more memory because all experts must be loaded."]),
    ]),
    Page("rag-practice", "hf.example", "/blog/rag-in-practice", "Retrieval-Augmented Generation in practice", "2024-03-10", [
        ("Overview", [
            "Retrieval-Augmented Generation grounds a Large Language Model in documents fetched from a Vector Database. A typical RAG pipeline chunks documents, embeds each chunk, retrieves the nearest chunks and passes them to the model.",
            "Hybrid retrieval combines BM25 with dense vectors and often improves recall, especially for exact technical terms such as model names."]),
        ("Example", [("pre", "chunks = split(documents)\nvectors = embed(chunks)\nhits = index.search(embed(question), k=5)")]),
    ]),
    Page("peft-docs", "hf-docs.example", "/docs/peft/lora", "PEFT: LoRA", "2024-02-01", [
        ("Overview", [
            "The PEFT library supports LoRA for parameter-efficient Fine-tuning. LoRA adds small trainable low-rank matrices to a frozen model, which lowers memory use and keeps the adapter files small.",
            "LoRA adapters can be merged back into the base model after training."]),
    ]),
    Page("techwire-open", "techwire.example", "/2024/07/open-weights-momentum", "Llama 3.1 signals open-weight momentum", "2024-07-25", [
        ("Analysis", [
            "Meta released Llama 3.1 405B, the largest open-weight model to date, TechWire reported. Analysts say Mistral AI, DeepSeek and Alibaba are also pushing open-weight releases while OpenAI and Anthropic keep their flagship models closed.",
            "Open-weight models are narrowing the gap with closed systems on benchmarks such as MMLU, according to several analysts."]),
    ]),
    Page("techwire-llama4", "techwire.example", "/2025/04/llama-4-arrives", "Meta ships Llama 4", "2025-04-06", [
        ("Report", [
            "Meta released Llama 4 Scout and Llama 4 Maverick over the weekend, TechWire reported. Llama 4 uses a Mixture of Experts architecture, a design Mistral AI and DeepSeek adopted earlier.",
            "Meta said Llama 4 Maverick outperforms GPT-4o on several multimodal benchmarks."]),
    ], first_round=2),
    Page("forum-open", "forum.example", "/t/best-open-model", "Best open model right now?", "2025-02-02", [
        ("Thread", ["In my experience DeepSeek-R1 beats Llama 3.1 on math prompts, but Llama 3.1 handles long documents better. Anyone else seeing this?"]),
    ]),
]

# Round 2: edited pages (new content hash => new version) and removed pages.
UPDATES: dict[str, Page] = {
    "llama-hub": Page("llama-hub", "meta-ai.example", "/models/llama", "The Llama model family", "2024-04-18", [
        ("Overview", ["Meta develops the Llama family of open-weight models. Llama 2, Llama 3, Llama 3.1 and Llama 4 are available for download."]),
        ("Current models", [_ul("Llama 3.1: 405B flagship, 128K context", "Llama 4 Scout: 10M token context window",
                                 "Llama 4 Maverick: 128 experts")]),
        ("Archive", [_ul("Llama 2: previous generation", "Llama 3: superseded by Llama 3.1")]),
    ], updated="2025-04-05", bare_div=True),
    "mistral-about": Page("mistral-about", "mistral.example", "/company", "About Mistral AI", "2023-09-27", [
        ("Company", ["Mistral AI is a Paris-based company building open-weight models. Mistral AI develops Mistral 7B and Mixtral. The company was founded in 2023 and releases many of its models under permissive licenses."]),
        ("Partnerships", ["In February 2024 Mistral AI partnered with Microsoft to distribute its models on Azure."]),
    ], updated="2024-02-26"),
}
REMOVED_IN_ROUND_2 = {"forum-open"}


def _render_block(b) -> str:
    if isinstance(b, str):
        return f"<p>{html.escape(b)}</p>"
    kind, payload = b
    if kind == "ul":
        return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in payload) + "</ul>"
    if kind == "table":
        rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in payload)
        return f"<table>{rows}</table>"
    return f"<pre><code>{html.escape(payload)}</code></pre>"


def render_page(p: Page, links_extra: list[tuple[str, str]] | None = None) -> str:
    body = f"<h1>{html.escape(p.title)}</h1>\n"
    for heading, blocks in p.sections:
        body += f"<h2>{html.escape(heading)}</h2>\n" + "\n".join(_render_block(b) for b in blocks) + "\n"
    if links_extra:
        body += "<ul>" + "".join(f'<li><a href="{h}">{html.escape(t)}</a></li>' for h, t in links_extra) + "</ul>"
    ld = ""
    if p.jsonld:
        ld = (f'<script type="application/ld+json">{{"@type":"Article","headline":"{html.escape(p.title)}","datePublished":"{p.published}"}}</script>')
    meta = f'<meta property="og:title" content="{html.escape(p.title)} | {HOSTS[p.host][0]}">'
    meta += f'<meta property="article:published_time" content="{p.published}T09:00:00Z">'
    if p.updated:
        meta += f'<meta property="article:modified_time" content="{p.updated}T09:00:00Z">'
    chrome_top = ('<header class="site-header"><nav><a href="/">Home</a><a href="/about-us">About</a>'
                  '<a href="/careers">Careers</a></nav></header><div class="cookie-banner">We use cookies. '
                  '<button>Accept</button></div>')
    chrome_bottom = ('<aside class="related"><a href="/popular">Popular posts</a></aside>'
                     '<footer><a href="/privacy">Privacy</a> <a href="https://twitter.example/share">Share</a>'
                     '<p>© 2026 Example Corp. All rights reserved.</p></footer>'
                     '<script>window.analytics = {track: function(){}}; // tracking</script>')
    main = f'<div id="content">{body}</div>' if p.bare_div else f"<main><article>{body}</article></main>"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(p.title)}</title>'
            f"{meta}{ld}<style>body{{font-family:sans-serif}}</style></head><body>{chrome_top}{main}{chrome_bottom}</body></html>")


def render_index(host: str, pages: list[Page]) -> str:
    name = HOSTS[host][0]
    items = [(p.path, p.title) for p in pages]
    extra = [("/private/internal-roadmap", "Internal roadmap")]
    if host == "nvidia.example":
        extra.append(("/drafts/unreleased-roadmap", "Unreleased roadmap"))
    index = Page("index", host, "/", f"{name} newsroom", "2024-01-01",
                 [("Latest posts", [f"Announcements and research from {name}."])])
    return render_page(index, items + extra + [("https://twitter.example/aihandle", "Follow us")])


@dataclass
class FixtureSite:
    """Serves the fixture web. `round` switches between the initial crawl and the later update."""

    round: int = 1
    flaky: dict[str, int] = field(default_factory=dict)  # url -> number of 503s before success
    hits: dict[str, int] = field(default_factory=dict)

    def pages(self) -> list[Page]:
        out = []
        for p in PAGES:
            if p.first_round > self.round:
                continue
            if self.round >= 2 and p.key in REMOVED_IN_ROUND_2:
                continue
            out.append(UPDATES[p.key] if self.round >= 2 and p.key in UPDATES else p)
        return out

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url, host, path = str(request.url), request.url.host, request.url.path or "/"
        self.hits[url.rstrip("/")] = self.hits.get(url.rstrip("/"), 0) + 1
        if host not in HOSTS:
            return httpx.Response(404, text="unknown host")
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\nDisallow: /drafts/\nCrawl-delay: 0\n")
        if self.flaky.get(url, 0) > 0:
            self.flaky[url] -= 1
            return httpx.Response(503, headers={"Retry-After": "0"}, text="unavailable")
        host_pages = [p for p in self.pages() if p.host == host]
        if path == "/":
            body = render_index(host, host_pages)
        else:
            match = next((p for p in host_pages if p.path == path), None)
            if match is None:
                return httpx.Response(404, text="not found")
            body = render_page(match)
        etag = '"' + hashlib.md5(body.encode()).hexdigest() + '"'
        if request.headers.get("If-None-Match") == etag:
            return httpx.Response(304, headers={"ETag": etag})
        return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8", "ETag": etag})


def fixture_sources() -> list[Source]:
    out = []
    for host, (name, stype, prio) in HOSTS.items():
        out.append(Source(
            source_id=host.split(".")[0], name=name, base_url=f"https://{host}/", source_type=str(stype),
            seed_urls=[f"https://{host}/"], allowed_domains=[host], priority=prio, crawl_frequency_s=3600, max_depth=2,
        ))
    return out


def page_by_key(key: str) -> Page:
    return next(p for p in PAGES if p.key == key)


def doc_key_for_url(url: str) -> str | None:
    for p in PAGES:
        if p.url.rstrip("/") == url.rstrip("/"):
            return p.key
    return None
