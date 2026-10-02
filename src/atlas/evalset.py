"""Gold evaluation set over the fixture corpus (after both crawl rounds).

relevant_docs  : fixture page keys that contain the answer
relevant_phrases: optional chunk-level constraint (a relevant chunk must contain one of these, case-insensitive)
key_points     : each entry is 'alt1|alt2'; an answer covers it if ANY alternative appears (completeness)
abstain        : the corpus does not contain the answer; the right behaviour is to say so
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalItem:
    id: str
    question: str
    type: str
    expected: str
    relevant_docs: list[str]
    key_points: list[str]
    relevant_phrases: list[str] = field(default_factory=list)
    abstain: bool = False


EVAL_SET: list[EvalItem] = [
    EvalItem("f1", "When did Meta release Llama 3?", "FACTUAL", "April 18, 2024", ["llama-3"], ["April 18, 2024"], ["April 18, 2024"]),
    EvalItem("f2", "What is the context window of Llama 4 Scout?", "FACTUAL", "10 million tokens", ["llama-4"], ["10 million|10M"], ["10 million", "10M"]),
    EvalItem("f3", "How many parameters does Mistral 7B have?", "FACTUAL", "7.3 billion", ["mistral-7b"], ["7.3 billion"], ["7.3 billion"]),
    EvalItem("f4", "Which license does Mixtral use?", "FACTUAL", "Apache 2.0", ["mixtral"], ["Apache 2.0"], ["Apache 2.0"]),
    EvalItem("f5", "What is the Model Context Protocol?", "FACTUAL", "An open standard connecting AI assistants to data and tools", ["mcp"], ["open standard"]),
    EvalItem("f6", "What does LoRA do?", "FACTUAL", "Freezes weights and trains low-rank matrices", ["lora", "peft-docs"], ["low-rank"]),
    EvalItem("f7", "Who introduced the Transformer architecture?", "FACTUAL", "Google researchers, in Attention Is All You Need", ["attention"], ["Attention Is All You Need|Google"]),
    EvalItem("f8", "What is Claude Code?", "FACTUAL", "An agentic coding tool in the terminal", ["claude-code"], ["agentic coding tool"]),
    EvalItem("r1", "Who developed Mixtral?", "RELATIONAL", "Mistral AI", ["mixtral", "mistral-about"], ["Mistral AI"]),
    EvalItem("r2", "Which cloud providers is NVIDIA working with on Blackwell?", "RELATIONAL", "AWS, Google, Microsoft, Meta", ["blackwell", "nvidia-google"],
             ["Amazon Web Services", "Google", "Microsoft"]),
    EvalItem("r3", "Which companies partnered with Microsoft?", "RELATIONAL", "OpenAI, Meta, Mistral AI", ["openai-microsoft", "llama-2", "mistral-about"],
             ["OpenAI", "Meta", "Mistral"]),
    EvalItem("r4", "Which model is DeepSeek-R1 based on?", "RELATIONAL", "DeepSeek-V3", ["deepseek-r1"], ["DeepSeek-V3"]),
    EvalItem("m1", "Which companies partner with NVIDIA and also build AI accelerators?", "MULTI_HOP", "Microsoft, AWS, Google, AMD",
             ["blackwell", "maia", "trainium2", "tpu-v5p", "mi300x"], ["Microsoft", "Amazon|AWS", "Google"]),
    EvalItem("m2", "Which open-weight models use a mixture of experts architecture?", "MULTI_HOP", "Mixtral, Llama 4, DeepSeek-V3",
             ["mixtral", "llama-4", "deepseek-v3", "moe-explained"], ["Mixtral", "Llama 4", "DeepSeek-V3"]),
    EvalItem("c1", "Compare Llama 3.1 and Mistral 7B", "COMPARATIVE", "405B/128K vs 7.3B/Apache 2.0", ["llama-3-1", "mistral-7b"],
             ["405B|128K", "7.3 billion|Apache"]),
    EvalItem("c2", "Compare Gemma and Gemini", "COMPARATIVE", "Gemma: lightweight open; Gemini: multimodal flagship", ["gemma", "gemini"],
             ["open", "multimodal"]),
    EvalItem("t1", "How has Meta's Llama family evolved since 2023?", "TEMPORAL", "Llama 2 → 3 → 3.1 → 4", ["llama-2", "llama-3", "llama-3-1", "llama-4", "llama-hub"],
             ["Llama 2", "Llama 3", "Llama 4"]),
    EvalItem("t2", "How has the open-source LLM landscape changed since 2023, which companies are driving it, and what architectural trends emerged?",
             "TEMPORAL", "Meta, Mistral, DeepSeek; MoE", ["llama-2", "llama-3", "llama-3-1", "llama-4", "mixtral", "mistral-7b", "deepseek-v3", "gemma", "moe-explained", "techwire-open"],
             ["Meta", "Mistral", "Mixture of Experts|experts"]),
    EvalItem("t3", "What is the latest model released by DeepSeek?", "TEMPORAL", "DeepSeek-R1", ["deepseek-r1"], ["DeepSeek-R1"]),
    EvalItem("a1", "Why do Mixture of Experts models need less compute per token?", "ANALYTICAL", "Only a few experts are active per token", ["moe-explained", "deepseek-v3"],
             ["small number of experts|fraction of the parameters|router|gating|activated per token"]),
    EvalItem("a2", "What are the trade-offs of Mixture of Experts models?", "ANALYTICAL", "Faster pretraining/inference but more memory", ["moe-explained"], ["memory"]),
    EvalItem("a3", "How does hybrid retrieval improve RAG?", "ANALYTICAL", "BM25 + dense improves recall", ["rag-practice"], ["BM25", "dense", "recall"]),
    EvalItem("e1", "Give an overview of AI accelerator announcements in late 2023", "EXPLORATORY", "Trainium2, Maia 100, TPU v5p, MI300X",
             ["tpu-v5p", "trainium2", "maia", "mi300x"], ["Trainium2", "Maia 100", "TPU v5p|Instinct MI300X"]),
    EvalItem("d1", "What architecture does DeepSeek-V3 use?", "FACTUAL", "Mixture of Experts with Multi-head Latent Attention", ["deepseek-v3"],
             ["Mixture of Experts|Multi-head Latent Attention"]),
    EvalItem("n1", "What is the parameter count of GPT-5?", "FACTUAL", "(not in corpus)", [], [], abstain=True),
    EvalItem("n2", "Which company acquired Hugging Face?", "RELATIONAL", "(not in corpus)", [], [], abstain=True),
]
