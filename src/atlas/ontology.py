"""The AI/technology ontology: seed entities (with aliases) and the relation schema."""

from __future__ import annotations

from .domain import EntityType as T

C, P, M, PR, TE, FW, PA, DS, BM, OR, HW = (
    T.COMPANY, T.PERSON, T.MODEL, T.PRODUCT, T.TECHNOLOGY, T.FRAMEWORK, T.PAPER, T.DATASET,
    T.BENCHMARK, T.ORGANIZATION, T.HARDWARE,
)

# (canonical name, type, description, aliases)
SEED_ENTITIES: list[tuple[str, T, str, list[str]]] = [
    # companies / orgs
    ("OpenAI", C, "AI research and deployment company", ["OpenAI Inc.", "OpenAI, Inc."]),
    ("Anthropic", C, "AI safety company, maker of Claude", ["Anthropic PBC"]),
    ("Google", C, "Technology company; parent of Google DeepMind", ["Alphabet", "Google LLC", "Google Research"]),
    ("Google DeepMind", C, "Google's AI research lab", ["DeepMind"]),
    ("Meta", C, "Technology company behind Llama", ["Meta AI", "Meta Platforms", "Facebook AI Research", "FAIR"]),
    ("NVIDIA", C, "GPU and AI computing company", ["Nvidia Corporation", "Nvidia"]),
    ("Microsoft", C, "Technology company; OpenAI partner", ["Microsoft Corporation", "Microsoft Research"]),
    ("Amazon Web Services", C, "Amazon's cloud division", ["AWS", "Amazon"]),
    ("AMD", C, "Semiconductor company", ["Advanced Micro Devices"]),
    ("Mistral AI", C, "French open-weight model developer", ["Mistral"]),
    ("DeepSeek", C, "Chinese AI lab", ["DeepSeek AI"]),
    ("Alibaba", C, "Chinese technology company behind Qwen", ["Alibaba Cloud"]),
    ("Hugging Face", C, "Open model hub and platform", ["HuggingFace"]),
    ("xAI", C, "AI company founded by Elon Musk", []),
    ("Stanford University", OR, "Research university", ["Stanford"]),
    # models and families
    ("GPT-4", M, "OpenAI multimodal large language model", ["GPT 4", "GPT4"]),
    ("GPT-4o", M, "OpenAI omni model across text, audio and vision", ["GPT 4o"]),
    ("GPT-5", M, "OpenAI model", ["GPT 5"]),
    ("Llama", M, "Meta's open-weight model family", ["LLaMA"]),
    ("Llama 2", M, "Meta open-weight LLM, July 2023", ["Llama-2", "LLaMA 2"]),
    ("Llama 3", M, "Meta open-weight LLM, April 2024", ["Llama-3", "Meta Llama 3"]),
    ("Llama 3.1", M, "Meta open-weight LLM including 405B", ["Llama-3.1"]),
    ("Llama 4", M, "Meta natively multimodal MoE models", ["Llama-4"]),
    ("Llama 4 Scout", M, "Llama 4 model with 10M context", []),
    ("Llama 4 Maverick", M, "Llama 4 model with 128 experts", []),
    ("Claude", M, "Anthropic's model family", []),
    ("Claude 3", M, "Anthropic model family: Haiku, Sonnet, Opus", ["Claude 3 family"]),
    ("Claude 3 Opus", M, "Most capable Claude 3 model", []),
    ("Claude 3 Sonnet", M, "Balanced Claude 3 model", []),
    ("Claude 3 Haiku", M, "Fastest Claude 3 model", []),
    ("GPT", M, "OpenAI's GPT model family", []),
    ("Gemini", M, "Google DeepMind multimodal model family", []),
    ("Gemini Ultra", M, "Largest Gemini model", []),
    ("Gemini Pro", M, "Mid-size Gemini model", []),
    ("Gemini Nano", M, "On-device Gemini model", []),
    ("Gemma", M, "Google open-weight models", []),
    ("Mistral 7B", M, "Mistral AI 7B open model", ["Mistral-7B"]),
    ("Mixtral", M, "Mistral AI sparse mixture-of-experts model", ["Mixtral 8x7B", "Mixtral of Experts"]),
    ("DeepSeek-V3", M, "671B-parameter MoE model", ["DeepSeek V3"]),
    ("DeepSeek-R1", M, "Open reasoning model trained with reinforcement learning", ["DeepSeek R1"]),
    ("Qwen2.5", M, "Alibaba open-weight model family", ["Qwen 2.5"]),
    ("BERT", M, "Bidirectional encoder model from Google", []),
    # products
    ("ChatGPT", PR, "OpenAI's chat assistant", []),
    ("Claude Code", PR, "Anthropic's agentic coding tool", []),
    ("GitHub Copilot", PR, "AI pair programmer", ["Copilot"]),
    ("Azure", PR, "Microsoft cloud platform", ["Microsoft Azure"]),
    ("Google Cloud", PR, "Google's cloud platform", ["Google Cloud Platform", "GCP"]),
    # hardware
    ("Blackwell", HW, "NVIDIA GPU architecture", ["NVIDIA Blackwell", "B200"]),
    ("Hopper", HW, "NVIDIA GPU architecture", ["H100", "NVIDIA H100"]),
    ("TPU", HW, "Google's tensor processing unit", ["Tensor Processing Unit", "Cloud TPU"]),
    ("TPU v5p", HW, "Google's TPU generation announced December 2023", ["Cloud TPU v5p"]),
    ("Trainium", HW, "AWS AI training chip", ["AWS Trainium"]),
    ("Trainium2", HW, "AWS second-generation training chip", ["AWS Trainium2"]),
    ("Maia 100", HW, "Microsoft's AI accelerator", ["Microsoft Maia 100", "Maia"]),
    ("Instinct MI300X", HW, "AMD data-center AI accelerator", ["MI300X", "AMD Instinct MI300X"]),
    # technologies / concepts
    ("Transformer", TE, "Attention-based neural architecture", ["Transformers", "transformer architecture"]),
    ("Mixture of Experts", TE, "Sparse architecture routing tokens to expert sub-networks",
     ["MoE", "Mixture-of-Experts", "sparse mixture-of-experts", "sparse mixture of experts"]),
    ("Retrieval-Augmented Generation", TE, "Grounding generation in retrieved documents", ["RAG"]),
    ("LoRA", TE, "Low-rank adaptation for parameter-efficient fine-tuning", ["Low-Rank Adaptation"]),
    ("Grouped-Query Attention", TE, "Attention variant sharing key/value heads", ["GQA", "grouped query attention"]),
    ("Sliding Window Attention", TE, "Attention restricted to a local window", ["SWA"]),
    ("Multi-head Latent Attention", TE, "DeepSeek's KV-cache-compressing attention", ["MLA"]),
    ("Model Context Protocol", TE, "Open protocol connecting AI assistants to tools and data", ["MCP"]),
    ("Reinforcement Learning", TE, "Learning from reward signals", ["RL"]),
    ("Agentic AI", TE, "AI systems that plan and act with tools", ["AI agents", "agentic systems"]),
    ("Vector Database", TE, "Database for embedding similarity search", ["vector databases", "vector store"]),
    ("Multimodal", TE, "Models handling several modalities", ["multimodality", "natively multimodal"]),
    ("Open-weight models", TE, "Models whose weights are publicly released",
     ["open weights", "open-weight", "open-source LLM", "open-source LLMs", "open-source models", "open-source model"]),
    ("AI accelerator", TE, "Chip specialised for AI workloads", ["AI accelerators", "AI chip", "AI chips", "accelerator"]),
    ("GPU", TE, "Graphics processing unit", ["GPUs"]),
    ("Large Language Model", TE, "Large-scale language model", ["LLM", "LLMs", "large language models"]),
    ("Fine-tuning", TE, "Adapting a pretrained model", ["fine-tuned", "finetuning"]),
    # papers
    ("Attention Is All You Need", PA, "2017 paper introducing the Transformer", []),
    ("LoRA: Low-Rank Adaptation of Large Language Models", PA, "2021 paper introducing LoRA", ["LoRA paper"]),
    ("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", PA, "2020 RAG paper", ["RAG paper"]),
    # datasets / benchmarks
    ("MMLU", BM, "Massive Multitask Language Understanding", []),
    ("HumanEval", BM, "Code generation benchmark", []),
    ("GSM8K", BM, "Grade-school math benchmark", []),
    ("MATH", BM, "Competition mathematics benchmark", []),
    ("LMArena", BM, "Crowdsourced model arena", ["Chatbot Arena"]),
]

# Entities that act as categories (targets of IS_A) rather than specific things.
CATEGORY_ENTITIES = {"AI accelerator", "Large Language Model", "Open-weight models", "GPU"}

# Words that begin model names; used to discover unseen versions (e.g. "Llama 3.2") as provisional MODELs.
MODEL_FAMILIES = ["GPT", "Llama", "Claude", "Gemini", "Gemma", "Mistral", "Mixtral", "Qwen", "DeepSeek", "Phi", "Grok", "Falcon"]
FAMILY_ENTITY = {"GPT": "GPT","Llama": "Llama", "Claude": "Claude", "Gemini": "Gemini", "Gemma": "Gemma",
                 "Mistral": "Mistral AI", "Mixtral": "Mixtral", "DeepSeek": "DeepSeek"}

ORG_TYPES = {C, OR}
CREATABLE = {M, PR, TE, FW, DS, BM, HW, PA}

# relation -> (allowed source types, allowed target types)
RELATIONS: dict[str, tuple[set[T], set[T]]] = {
    "RELEASED": (ORG_TYPES, {M, PR, FW, DS, BM, TE, HW, PA}),
    "DEVELOPS": (ORG_TYPES, {M, PR, TE, FW, HW}),
    "ACQUIRED": (ORG_TYPES, ORG_TYPES),
    "PARTNERS_WITH": (ORG_TYPES, ORG_TYPES),
    "INVESTED_IN": (ORG_TYPES, ORG_TYPES),
    "USES": ({M, PR, FW, HW}, {TE, M}),
    "BASED_ON": ({M, TE, PR}, {M, TE}),
    "TRAINED_ON": ({M}, {DS, TE}),
    "EVALUATED_ON": ({M}, {BM}),
    "OUTPERFORMS": ({M, HW}, {M, HW}),
    "INTRODUCED": ({PA, C, OR}, {TE, M}),
    "IS_A": ({M, HW, PR, TE}, {TE}),
    "VARIANT_OF": ({M}, {M}),
}
SYMMETRIC = {"PARTNERS_WITH"}
REL_LABEL = {
    "RELEASED": "released", "DEVELOPS": "develops", "ACQUIRED": "acquired", "PARTNERS_WITH": "partners with",
    "INVESTED_IN": "invested in", "USES": "uses", "BASED_ON": "is based on", "TRAINED_ON": "trained on",
    "EVALUATED_ON": "evaluated on", "OUTPERFORMS": "outperforms", "INTRODUCED": "introduced",
    "IS_A": "is a", "VARIANT_OF": "is a variant of",
}
