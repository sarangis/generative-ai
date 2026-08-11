# Study Notes Index — Advanced Certification in Agentic & Generative AI

**IISc Bengaluru / TalentSprint** · Prof. Sashikumaar Ganesan · 27 weeks, 10 modules
Complete study notes for every module, built from the source lecture decks.

---

## The modules

| # | Module | Weeks | Notes | Lines | Diagrams |
|---|---|---|---|---|---|
| **M1** | [GenAI Foundations](M1/M1_Study_Notes.md) | 1–2 | Token prediction, language models, scaling laws, foundation models | 1,701 | 11 |
| **M2** | [Core Concepts, Transformers & Prompt Engineering](M2/M2_Study_Notes.md) | 3–4 | Self-attention, MHA, positional encoding, embeddings, the full prompting spectrum | 1,738 | 12 |
| **M3** | [LLMs: APIs, Architecture & Tokenization](M3/M3_Study_Notes.md) | 5–6 | Every API parameter from first principles, BPE/WordPiece/SentencePiece, LM head | 1,225 | 8 |
| **M4** | [Multimodal AI](M4/M4_Study_Notes.md) | 7–8 | ViT, CLIP/SigLIP, VLM connectors, audio & ASR/TTS, fusion, the $300 LLaVA recipe | 1,002 | 8 |
| **M5** | [Agentic AI](M5/M5_Study_Notes.md) | 9–10 | Agent anatomy, ReAct, tool use, planning, cognitive architectures, RAG | 1,172 | 11 |
| **M6** | [Scientific ML & PINNs](M6/M6_Study_Notes.md) ⚠️ | 11–12 | PDEs, physics-informed loss, neural operators, neurosymbolic | 557 | 5 |
| **M7** | [Fine-Tuning & RLHF](M7/M7_Study_Notes.md) | 13–14 | LoRA/QLoRA, catastrophic forgetting, instruction tuning, RLHF, DPO | 1,183 | 10 |
| **M8** | [AI Safety & Evaluation](M8/M8_Study_Notes.md) | 15–16 | Evaluation layers, guardrails, prompt injection, red teaming, fairness, hallucination | 1,182 | 9 |
| **M9** | [LLMOps](M9/M9_Study_Notes.md) | 17–19 | MCP, A2A, infrastructure, quantisation, vLLM serving, scaling laws, cost | 1,039 | 9 |
| **M10** | [Federated & Distributed AI](M10/M10_Study_Notes.md) | 20–21 | FedAvg, differential privacy, DDP/ZeRO/pipeline, Docker, HF Spaces, **capstone** | 877 | 10 |

**Total: 11,676 lines · 93 Mermaid diagrams · 611 KB**

> ⚠️ **M6 caveat:** its folder contains only handwritten scanned notes with no extractable slide text. The notes are assembled from the decodable equations plus the curriculum spec and designated references, with every section labelled 📝 (sourced) / 📋 (spec) / 📚 (reference). **Revise once real M6 slides are available.**

---

## Each file contains

- **Module map** — every source PDF mapped to its concept ID and topic
- **Visual atlas** — a Mermaid mind map plus correlation diagrams (marked ⭐ where they carry the module's central idea)
- **Lecture-by-lecture notes** — formulas written out, worked examples computed
- **Master misconceptions table** — consolidated from every deck's "myths vs reality" slides
- **Glossary** — 30–50 terms per module
- **References** — books, papers, and tools mapped *per lecture*, not as a generic booklist
- **Self-check question bank** — 35–55 questions per module

---

## Rendering the diagrams

Mermaid renders natively in **GitHub**, **VS Code** (Markdown Preview Mermaid extension), **Obsidian**, and **Notion**. In a plain text editor you'll see the source.

---

## The five diagrams worth studying first

| Diagram | Where | Why |
|---|---|---|
| **The master pipeline** | [M1 §D](M1/M1_Study_Notes.md) | Shows M1's three lectures as one machine — logits → temperature → softmax → decode → loop |
| **Inside one Transformer block** | [M2 §C](M2/M2_Study_Notes.md) | Every §2.2 component in one picture; ends exactly where M1 §1.2.2 began |
| **Every modality becomes tokens** | [M4 §B](M4/M4_Study_Notes.md) | Vision, audio and text front-ends converging on one unchanged Transformer |
| **The ReAct trace, annotated** | [M5 §D](M5/M5_Study_Notes.md) | With trace economics and the debug-backwards rule |
| **Production agentic architecture** | [M10 §J](M10/M10_Study_Notes.md) | Six layers, each labelled with the module that built it — **design your capstone against this** |

---

## Cross-module threads

Some ideas recur across modules and are worth tracing end to end:

| Thread | Path |
|---|---|
| **Hallucination** | M1 (named) → M2 (RAG grounding) → M4 (VLM hallucination) → M8 §8.6 (architectural root cause, detection, grounding spectrum) |
| **Goodhart's Law** | M7 §7.4.4 (corrupts the reward model) → M8 §8.1.1 (corrupts your evaluation dashboard) |
| **Attention cost** | M2 §2.2.2 ($O(n^2)$) → M3 §3.2.2 (Flash Attention fixes memory not time) → M9 §9.2.3 (KV cache, continuous batching) |
| **LoRA** | M2 §2.2.3 (which matrices) → M4 §4.3.4 (on VLMs) → M7 §7.2 (in full) → M9 (serving adapters) |
| **Tokens as cost** | M1 §1.2.1 → M3 §3.3.5 (the token economy) → M5 (agent trace economics) → M9 §9.7.6 (cost optimisation) |
| **The five-level framework** | M1 (introduced) → M2 Part A (a deck per level) → M10 final lecture (you entered at L1, leave at L4–L5) |

---

## Deadlines recorded in these notes

| Item | Date |
|---|---|
| M1 Assignment 1 | 1 Mar 2026 |
| Mini-Project 1 (Spam Classification) | 15 & 22 Mar 2026 |
| Mini-Project 2 (Incident Management) | 12 & 19 Apr 2026 |
| Mini-Project 3 (Agentic Banking Assistant) | M7 |
| Mini-Project 4 (Airline Support) — 10 pts, team | 7 & 14 Jun 2026 |
| Mini-Project 5 (Deploy to HF Spaces) — 10 pts, team | 28 Jun 2026 |
| Final lecture | 27 Jun 2026 |
| Capstone | Weeks 22+ (40 h) |

---

*Compiled from the source lecture decks in each module folder. Concept IDs preserved throughout for cross-referencing against the course knowledge graph.*
