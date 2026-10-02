"""Source registry for crawling the real web. Seeds are listing pages; link-following is limited to the source's domain.

These are starting points, not guarantees: sites change layout and some render client-side (a plain HTTP crawler
only sees server-rendered HTML). Crawling honours robots.txt and identifies itself via the User-Agent in Settings.
"""

from __future__ import annotations

from .domain import Source, SourceType

LIVE_SOURCES: list[Source] = [
    Source("hf-blog", "Hugging Face Blog", "https://huggingface.co/blog", SourceType.TECHNICAL_BLOG,
           seed_urls=["https://huggingface.co/blog"], allowed_domains=["huggingface.co"], include_patterns=[r"/blog/[^/]+(/[^/]+)?$"], exclude_patterns=[r"/blog/(feed\.xml|community)$"],
           priority=6, crawl_frequency_s=6 * 3600, max_depth=1),
    Source("openai-news", "OpenAI", "https://openai.com/news/", SourceType.OFFICIAL,
           seed_urls=["https://openai.com/news/"], allowed_domains=["openai.com"], include_patterns=[r"/(index|news)/[^/]+/?$"],
           priority=9, crawl_frequency_s=6 * 3600, max_depth=1),
    Source("anthropic-news", "Anthropic", "https://www.anthropic.com/news", SourceType.OFFICIAL,
           seed_urls=["https://www.anthropic.com/news"], allowed_domains=["anthropic.com"], include_patterns=[r"/news/[^/]+$"],
           priority=9, crawl_frequency_s=6 * 3600, max_depth=1),
    Source("meta-ai", "Meta AI", "https://ai.meta.com/blog/", SourceType.OFFICIAL,
           seed_urls=["https://ai.meta.com/blog/"], allowed_domains=["ai.meta.com"], include_patterns=[r"/blog/[^/]+/?$"],
           priority=9, crawl_frequency_s=12 * 3600, max_depth=1),
    Source("deepmind-blog", "Google DeepMind", "https://deepmind.google/discover/blog/", SourceType.OFFICIAL,
           seed_urls=["https://deepmind.google/discover/blog/"], allowed_domains=["deepmind.google"], include_patterns=[r"/blog/[^/]+/?$"],
           priority=9, crawl_frequency_s=12 * 3600, max_depth=1),
    Source("nvidia-blog", "NVIDIA Technical Blog", "https://developer.nvidia.com/blog/", SourceType.TECHNICAL_BLOG,
           seed_urls=["https://developer.nvidia.com/blog/"], allowed_domains=["developer.nvidia.com"], include_patterns=[r"/blog/[^/]+/?$"],
           priority=7, crawl_frequency_s=12 * 3600, max_depth=1),
    Source("mistral-news", "Mistral AI", "https://mistral.ai/news/", SourceType.OFFICIAL,
           seed_urls=["https://mistral.ai/news/"], allowed_domains=["mistral.ai"], include_patterns=[r"/news/[^/]+/?$"],
           priority=8, crawl_frequency_s=12 * 3600, max_depth=1),
]
