# Module 5 — Agentic AI: Agents, ReAct, Planning & RAG · Study Notes

**Programme:** Advanced Certification in Agentic and Generative AI
**Institution:** IISc Bengaluru / TalentSprint · **Instructor:** Prof. Sashikumaar Ganesan
**Module duration:** 6 hours (Weeks 9–10) · **Prerequisite:** M1–M4

> **What this module is really about.** Everything before this produced *text*. Module 5 produces *actions*. The pivot is one sentence from the very first deck:
>
> ### 🔑 **A chatbot responds. An agent solves. The loop is what makes the difference.**
>
> This is the module the whole programme is named after, and it is where the course's economics get real: nearly every deck quotes a **cost per query**. Agentic AI is an engineering discipline with a P&L, not a research demo.

---

## Table of Contents

1. [Module map](#0-module-map)
2. [🗺️ Visual atlas](#-visual-atlas--mind-map--correlation-diagrams)
3. **5.1 Agent Fundamentals**
   - [5.1.1 What is an AI Agent?](#511--what-is-an-ai-agent)
   - [5.1.2 Agent Anatomy](#512--agent-anatomy-perception-decision-action)
   - [5.1.3 Simple Reactive Agents](#513--simple-reactive-agents)
   - [5.1.4 Agents with Memory](#514--agents-with-memory)
   - [5.1.5 Agent Goals and Objectives](#515--agent-goals-and-objectives)
   - [5.1.6 LLMs as Agent Brains](#516--llms-as-agent-brains)
   - [5.1.7 SLMs as Lightweight Agent Engines](#517--slms-as-lightweight-agent-engines)
   - [5.1.8 Building a Simple Agent (lab)](#518--building-a-simple-agent-coding-lab)
4. **5.2 The ReAct Framework**
   - [5.2.1 ReAct: Reasoning + Acting](#521--react-reasoning--acting)
   - [5.2.2 The Thought-Action-Observation Loop](#522--the-thought-action-observation-loop)
   - [5.2.3 Tool Use and Function Calling](#523--tool-use-and-function-calling)
   - [5.2.4 ReAct Loop Visualisation](#524--react-loop-visualisation)
   - [5.2.5 When to Reason vs. Act](#525--when-to-reason-vs-act)
   - [5.2.6 Self-Reflection (Reflexion)](#526--self-reflection-and-iterative-refinement)
   - [5.2.7 Implementing ReAct (lab)](#527--implementing-react-agents-coding-lab)
5. **5.3 Planning and Decomposition**
   - [5.3.1 Classical Planning](#531--classical-planning)
   - [5.3.2 LLM-Based Planning](#532--llm-based-planning)
   - [5.3.3 Hybrid Planning](#533--hybrid-planning-approaches)
   - [5.3.4 Task Decomposition](#534--task-decomposition)
   - [5.3.5 Goal-Directed Behaviour](#535--goal-directed-behaviour)
   - [5.3.6 Cognitive Architectures](#536--cognitive-architectures-loops-to-state-graphs)
   - [5.3.7 Planning System Implementation (lab)](#537--planning-system-implementation-coding-lab)
6. [5.4 RAG — Retrieval-Augmented Generation](#54--rag--retrieval-augmented-generation)
7. [Assignments & application](#assignments--application)
8. [Master list of misconceptions](#master-list-of-misconceptions)
9. [Glossary](#glossary)
10. [References](#references-and-further-study)
11. [Self-check question bank](#self-check-question-bank)

---

## 0. Module map

| File | Concept ID | Content |
|---|---|---|
| `C01_What_Is_An_AI_Agent.pdf` | `AIAGFNTH000001` | **5.1.1** What is an AI Agent? |
| `C02_Agent_Anatomy.pdf` | `AIAGFNTH000002` | **5.1.2** Agent Anatomy |
| `C03_Simple_Reactive_Agents.pdf` | `AIAGFNTH000003` | **5.1.3** Simple Reactive Agents |
| `C04_Agents_With_Memory.pdf` | `AIAGFNTH000004` | **5.1.4** Agents with Memory |
| `C05_Agent_Goals.pdf` | `AIAGFNTH000005` | **5.1.5** Agent Goals and Objectives |
| `C06_LLMs_As_Agent_Brains.pdf` | `AIAGFNTH000006` | **5.1.6** LLMs as Agent Brains |
| `C07_SLMs_Agent_Engines.pdf` | `AIAGFNTH000019` | **5.1.7** SLMs as Lightweight Agent Engines |
| `C08_Building_Simple_Agent.pdf` | `AIAGFNCD000001` | **5.1.8** Building a Simple Agent — **lab** |
| `C09_ReAct_Framework.pdf` | `AIAGINTH000001` | **5.2.1** ReAct Framework |
| `C10_TAO_Loop.pdf` | `AIAGINTH000002` | **5.2.2** Thought-Action-Observation Loop |
| `C11_Tool_Use.pdf` | `AIAGINTH000003` | **5.2.3** Tool Use and Function Calling |
| `C12_ReAct_Visualization.pdf` | `AIAGINVZ000001` | **5.2.4** ReAct Loop Visualisation |
| `C13_Reason_vs_Act.pdf` | `AIAGINTH000004` | **5.2.5** When to Reason vs. Act |
| `C14_Self_Reflection.pdf` | `AIAGINTH000019` | **5.2.6** Self-Reflection (Reflexion) |
| `C15_Implementing_ReAct.pdf` | `AIAGINCD000001` | **5.2.7** Implementing ReAct — **lab** |
| `C16_Classical_Planning.pdf` | `AIAGINTH000005` | **5.3.1** Classical Planning |
| `C17_LLM_Planning.pdf` | `AIAGINTH000006` | **5.3.2** LLM-Based Planning |
| `C18_Hybrid_Planning.pdf` | `AIAGINTH000007` | **5.3.3** Hybrid Planning |
| `C19_Task_Decomposition.pdf` | `AIAGINTH000008` | **5.3.4** Task Decomposition |
| `C20_Goal_Directed.pdf` | `AIAGINTH000009` | **5.3.5** Goal-Directed Behaviour |
| `C21_Cognitive_Arch.pdf` | `AIAGINTH000020` | **5.3.6** Cognitive Architectures |
| `C22_Planning_Implementation.pdf` | `AIAGINCD000002` | **5.3.7** Planning Implementation — **lab** |
| `M5_1_Agent_Fundamentals.pdf` | — | Consolidated 5.1 deck |
| `RAG_Content_covered_by_Mentor.zip` | — | **RAG session material** |
| `M5_AST_01_Leveraging_LLMs_for_Querying_Insights.ipynb` | — | **Assignment 1** |
| `M5_AST_02_Building_Agents_using_LangGraph_*.ipynb` | — | **Assignment 2** (with / without OpenAI API) |
| `M5_Additional_NB_01_Tool_Calling.ipynb` | — | Tool calling lab |
| `M5_Additional_NB_02_Building_Agent_using_LangChain.ipynb` | — | LangChain agent lab |
| `M5_Additional_NB_01/02_Building_RAG_Pipeline_using_LangChain_*.ipynb` | — | **RAG pipeline labs** |
| `Steps_for_Stock_Prices_Insights_Generation_Application.pdf` | — | Application walkthrough |
| `Application.zip` · `Session_Objectives.pdf` | — | Tutorial (18 Apr) |

> ℹ️ `AI-MM-IN-TH-000007`…`000013` in this folder are **duplicates of the M4 multimodal decks** — see [M4 notes](../M4/M4_Study_Notes.md).

---

# 🗺️ Visual atlas — mind map & correlation diagrams

## A. Module 5 mind map

```mermaid
mindmap
  root((MODULE 5 - Agentic AI))
    5.1 Fundamentals
      Why agents exist
        LLMs are stateless
        No tools
        Single turn
      Five capabilities
        Perception
        State management
        Reasoning
        Tool use
        Iteration
      Agent anatomy
        Perception
        Memory
        Reasoning LLM
        Action tools
      Agent types
        Reactive - no state
        Memory-augmented
        Goal-directed
      Memory
        In-context short-term
        External vector DB
      Goals
        Explicit
        Implicit
        Decomposed
      Brains
        LLMs as controllers
        SLMs tiered routing
    5.2 ReAct
      Reasoning plus Acting
      Thought Action Observation
      Tool use and function calling
        Define schemas
        LLM selects
        Orchestrator executes
        Result injected
      Reason vs Act balance
        Thought loops
        Action storms
      Reflexion
        Generate evaluate reflect retry
      Failure modes
        Hallucinated tools
        Infinite loops
        Wrong tool selection
    5.3 Planning
      Classical STRIPS
        Complete and sound
        Closed world
      LLM planning
        Plan-then-execute
        Interleaved
      Hybrid
        LLM then verifier
        Plan repair
      Task decomposition
        Recursive parallel sequential
        DAG dependencies
      Goal-directed behaviour
        Goal drift
        Re-inject goals
      Cognitive architectures
        Loop to state graph
        LangGraph AutoGen Swarm
    5.4 RAG
      Chunk embed store
      Retrieve and inject
      Vector databases
      Grounding
```

## B. ⭐ Why agents exist — the three limitations

```mermaid
flowchart TD
    L["<b>A standard LLM</b>"]
    L --> L1["❌ <b>STATELESS</b><br/>Turn 1: 'What flights BLR→BOM on Mar 15?'<br/>→ 'IndiGo 6:30 AM, Air India, SpiceJet…'<br/>Turn 2: 'Book the IndiGo one'<br/>→ <i>'I don't know which flight you mean.'</i><br/><br/>Each call is independent."]
    L --> L2["❌ <b>NO TOOLS</b><br/>Cannot browse the web, query a database,<br/>call an API, or execute code."]
    L --> L3["❌ <b>SINGLE TURN</b><br/>Cannot iterate toward a solution.<br/>One prompt, one answer, done."]

    L1 --> A["<b>AN AGENT</b><br/>solves all three"]
    L2 --> A
    L3 --> A

    A --> LOOP["<b>THE CORE LOOP</b><br/>Perceive → Reason → Act → Observe<br/>↻ repeat until the goal is met"]

    KEY["🔑 <b>A chatbot responds. An agent solves.</b><br/>The loop is what makes the difference."]
    LOOP -.-> KEY

    style L1 fill:#fce8e6,stroke:#c5221f,color:#000
    style L2 fill:#fce8e6,stroke:#c5221f,color:#000
    style L3 fill:#fce8e6,stroke:#c5221f,color:#000
    style A fill:#e6f4ea,stroke:#137333,stroke-width:2px,color:#000
    style LOOP fill:#e6f4ea,stroke:#137333,stroke-width:3px,color:#000
    style KEY fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#000
```

## C. Agent anatomy — the four-component loop

```mermaid
flowchart LR
    ENV(["🌍 Environment"]) -->|observations| P["<b>① PERCEPTION</b><br/>user input · API responses ·<br/>tool outputs · sensor data<br/><i>the agent's eyes and ears</i>"]
    P --> M["<b>② MEMORY</b><br/>short-term: context window<br/>long-term: vector DB<br/><i>what's been done, what remains</i>"]
    M --> R["<b>③ REASONING</b> 🧠<br/>the LLM interprets observations +<br/>memory, plans, decides the next action<br/><i>chain-of-thought</i>"]
    R --> A["<b>④ ACTION</b><br/>API calls · web search · code execution ·<br/>DB queries · file operations<br/><i>how the agent affects the world</i>"]
    A -->|effects| ENV
    A -.->|"⑤ <b>ITERATION</b><br/>loop until goal met —<br/>no human prompt needed"| P

    COST["💰 A full resolution costs <b>&lt; $0.001</b><br/>— orders of magnitude cheaper<br/>than a human agent."]
    A -.-> COST

    style P fill:#e8f0fe,stroke:#3367d6,stroke-width:2px,color:#000
    style M fill:#e6f4ea,stroke:#137333,stroke-width:2px,color:#000
    style R fill:#f3e8fd,stroke:#8430ce,stroke-width:3px,color:#000
    style A fill:#fff3e0,stroke:#e8710a,stroke-width:2px,color:#000
    style COST fill:#fef7e0,stroke:#f9ab00,stroke-dasharray: 5 5,color:#000
```

## D. ⭐ The ReAct trace — annotated

```mermaid
flowchart TD
    Q["<b>User goal:</b> 'Which had higher revenue in 2023,<br/>Company A or Company B?'"]

    Q --> T1["💭 <b>THOUGHT 1</b><br/><i>'I need revenue figures for both.<br/>Let me search for Company A first.'</i><br/>— model-generated reasoning"]
    T1 --> A1["⚙️ <b>ACTION 1</b><br/>search('Company A 2023 revenue')<br/>— structured tool call"]
    A1 --> O1["🔵 <b>OBSERVATION 1</b><br/>'Company A reported $4.2B'<br/>— <b>the ONLY real-world data</b>"]
    O1 --> T2["💭 <b>THOUGHT 2</b><br/><i>'Got A. Now I need B.'</i>"]
    T2 --> A2["⚙️ <b>ACTION 2</b><br/>search('Company B 2023 revenue')"]
    A2 --> O2["🔵 <b>OBSERVATION 2</b><br/>'Company B reported $3.8B'"]
    O2 --> T3["💭 <b>THOUGHT 3</b><br/><i>'4.2 &gt; 3.8. I can answer.'</i>"]
    T3 --> ANS(["✅ <b>FINAL ANSWER</b><br/>Company A, $4.2B vs $3.8B"])

    N1["📊 <b>Trace economics</b><br/>3 iterations ≈ <b>920 tokens</b> ≈ <b>$0.0006</b><br/>at GPT-4o-mini. Cheaper than a cup of chai.<br/>Budget ~200 tokens/iteration.<br/>Don't let iterations exceed 50% of the window."]
    N2["🐛 <b>Debug BACKWARDS</b><br/>Answer → Observation → Action → Thought.<br/>Start from the wrong answer, trace back<br/>through Observations to find the root cause."]
    N3["⚠️ <b>Observations are 25% of tokens.</b><br/>Truncate long tool results."]

    ANS -.-> N1
    ANS -.-> N2
    O2 -.-> N3

    style T1 fill:#f3e8fd,stroke:#8430ce,color:#000
    style T2 fill:#f3e8fd,stroke:#8430ce,color:#000
    style T3 fill:#f3e8fd,stroke:#8430ce,color:#000
    style A1 fill:#fff3e0,stroke:#e8710a,color:#000
    style A2 fill:#fff3e0,stroke:#e8710a,color:#000
    style O1 fill:#e8f0fe,stroke:#3367d6,stroke-width:2px,color:#000
    style O2 fill:#e8f0fe,stroke:#3367d6,stroke-width:2px,color:#000
    style ANS fill:#e6f4ea,stroke:#137333,stroke-width:3px,color:#000
    style N2 fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#000
```

## E. The three ReAct failure modes

```mermaid
flowchart TD
    F["<b>Agent trace looks wrong.</b><br/>Which failure is it?"]

    F --> F1["<b>① HALLUCINATED TOOL</b><br/>Agent calls <code>get_weather()</code><br/>but only <code>search()</code> exists<br/><br/>🛠️ <b>Fix:</b> validate tool name against<br/>the registry before execution;<br/>return a clear error to the agent"]
    F --> F2["<b>② INFINITE LOOP</b><br/>Agent searches the same thing<br/>5 times with different phrasings<br/><br/>🛠️ <b>Fix:</b> max-iteration budget,<br/>deduplicate observations,<br/>'act within 2 Thoughts' in the prompt"]
    F --> F3["<b>③ WRONG TOOL SELECTION</b><br/>Agent uses <code>search()</code> when<br/><code>calculator()</code> was correct<br/><br/>🛠️ <b>Fix:</b> improve tool <b>descriptions</b><br/>— include negative examples<br/>('do NOT use this for…')"]

    R["⭐ <b>Description quality determines<br/>selection accuracy.</b> The tool description<br/>is a prompt — engineer it like one."]
    F3 -.-> R

    style F1 fill:#fce8e6,stroke:#c5221f,color:#000
    style F2 fill:#fce8e6,stroke:#c5221f,color:#000
    style F3 fill:#fce8e6,stroke:#c5221f,color:#000
    style R fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#000
```

## F. Reason vs. Act — the meta-cognitive balance

```mermaid
flowchart TD
    Q["After every Observation:<br/><b>Do I have enough to answer,<br/>or do I need to think/act more?</b>"]

    Q -->|"too far this way"| L["❌ <b>THOUGHT LOOP</b><br/>Agent thinks and thinks<br/>but never acts.<br/><i>Burns tokens, produces nothing.</i>"]
    Q -->|"too far that way"| S["❌ <b>ACTION STORM</b><br/>Agent fires tools without planning.<br/>Random searches, irrelevant results.<br/><i>Burns API budget.</i>"]
    Q -->|"calibrated"| G["✅ <b>BALANCED</b>"]

    G --> P1["<b>Think-Once-Act-Once</b><br/>simple lookups"]
    G --> P2["<b>Interleaved ReAct</b><br/>general purpose"]
    G --> P3["<b>Plan-Then-Execute</b><br/>complex multi-step"]

    CAL["📏 <b>Calibrate by task tier</b><br/>Lookup → <b>2 iterations</b><br/>Analysis → <b>5–7 iterations</b><br/>Research → <b>10–15 iterations</b><br/><br/>📏 <b>Human heuristic</b><br/>If you'd decide in &lt;3 seconds → minimal Thought.<br/>If &gt;30 seconds → deep Thought.<br/><br/>🔒 <b>Enforce in the prompt:</b><br/>'Act within 2 Thoughts' · 'max N tool calls'"]
    G -.-> CAL

    KEY["🔑 The reason-vs-act balance is a<br/><b>design decision, not an emergent property.</b><br/>Engineer it deliberately."]
    Q -.-> KEY

    style L fill:#fce8e6,stroke:#c5221f,color:#000
    style S fill:#fce8e6,stroke:#c5221f,color:#000
    style G fill:#e6f4ea,stroke:#137333,stroke-width:3px,color:#000
    style CAL fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#000
    style KEY fill:#e8f0fe,stroke:#3367d6,stroke-width:2px,color:#000
```

## G. Reflexion — adding self-correction to ReAct

```mermaid
flowchart LR
    G1["<b>GENERATE</b><br/>produce an answer<br/>via ReAct"] --> E["<b>EVALUATE</b><br/>is it correct?"]
    E -->|"✅ pass"| DONE(["Return answer"])
    E -->|"❌ fail"| REF["<b>REFLECT</b><br/><i>verbal</i> self-critique:<br/>'The test failed because I<br/>assumed a sorted input.'"]
    REF --> RETRY["<b>RETRY</b><br/>regenerate using<br/>the reflection as context"]
    RETRY --> E

    EV["<b>Three evaluation strategies</b><br/>① Constraint validation — <b>$0</b><br/>&nbsp;&nbsp;&nbsp;(does the JSON parse? do tests pass?)<br/>② LLM-as-judge — <b>~$0.001</b><br/>③ Ground-truth comparison — when available"]
    E -.-> EV

    W["⚠️ <b>ALWAYS set a retry budget (max 3).</b><br/>Without one, reflection loops forever —<br/>the <b>#1 deployment risk</b> of this pattern."]
    RETRY -.-> W

    R["📈 Shinn et al. (2023): <b>+24% on code generation</b><br/>with verbal self-reflection.<br/>Use for high-stakes tasks where<br/><b>correctness matters more than speed</b>."]

    style REF fill:#f3e8fd,stroke:#8430ce,stroke-width:2px,color:#000
    style W fill:#fce8e6,stroke:#c5221f,stroke-width:3px,color:#000
    style EV fill:#e8f0fe,stroke:#3367d6,color:#000
    style R fill:#e6f4ea,stroke:#137333,color:#000
```

## H. Planning — three approaches

```mermaid
flowchart TD
    T{"What kind of task?"}

    T -->|"Fully specified world:<br/>robotics, games, logistics"| C["<b>CLASSICAL PLANNING</b><br/>state-space search + <b>STRIPS</b><br/>(preconditions + effects)<br/><br/>✅ <b>completeness + soundness guaranteed</b><br/>— if a plan exists, it's found, and it works<br/>Forward chaining (from start) or<br/>backward chaining (from goal)<br/><br/>❌ fails on open worlds and natural language"]

    T -->|"Natural-language goal,<br/>simple task"| L["<b>LLM-BASED PLANNING</b><br/>plan generated by prompting<br/>Plan-then-execute <b>or</b> interleaved<br/>CoT: 'think step by step' + structured output<br/><br/>✅ no formal logic needed<br/>❌ plans can be <b>plausible but infeasible</b>:<br/>ordering errors, budget violations,<br/>missing constraints"]

    T -->|"Constrained or<br/>safety-critical"| H["<b>HYBRID PLANNING</b> ⭐<br/>① LLM → verifier<br/>② classical skeleton → LLM fills details<br/>③ <b>iterative plan repair</b><br/><br/>💰 verification costs ~<b>$0.02 extra</b><br/>but <b>reduces execution failures by 60%</b><br/><br/><i>Devin exemplifies this: LLM plans code changes,<br/>test suite verifies, replan on failure</i>"]

    H --> REPAIR["🔑 <b>Plan repair is the key differentiator</b><br/>— adapt the plan when reality doesn't match<br/>expectations, <b>preserving completed work</b><br/>and replanning only the remainder."]

    style C fill:#e8f0fe,stroke:#3367d6,color:#000
    style L fill:#fff3e0,stroke:#e8710a,color:#000
    style H fill:#e6f4ea,stroke:#137333,stroke-width:3px,color:#000
    style REPAIR fill:#fef7e0,stroke:#f9ab00,stroke-width:2px,color:#000
```

## I. Cognitive architecture evolution

```mermaid
flowchart LR
    A1["<b>REACTIVE</b><br/>action = rules(percept)<br/>no state"] --> A2["<b>ReAct LOOP</b><br/>Thought→Action→Observation<br/>linear, single path"]
    A2 --> A3["<b>STATE MACHINE</b><br/>explicit states<br/>+ transitions"]
    A3 --> A4["<b>STATE GRAPH</b> ⭐<br/>nodes · conditional edges ·<br/><b>cycles</b> · persistence<br/><i>LangGraph</i>"]

    F["<b>Framework fit</b><br/><b>LangGraph</b> — explicit state graph; branching,<br/>persistence, checkpointing<br/><b>AutoGen</b> — conversation patterns<br/><b>Swarm</b> — lightweight handoff"]
    A4 -.-> F

    R["📌 <b>Start with ReAct.</b> Upgrade to LangGraph<br/>when you need <b>branching, persistence,<br/>or checkpointing</b>.<br/><br/>🔑 Agent control flow is a <b>designable artifact</b><br/>— not a fixed loop, but a graph you architect."]
    A4 -.-> R

    style A1 fill:#fce8e6,stroke:#c5221f,color:#000
    style A2 fill:#fff3e0,stroke:#e8710a,color:#000
    style A3 fill:#fef7e0,stroke:#f9ab00,color:#000
    style A4 fill:#e6f4ea,stroke:#137333,stroke-width:3px,color:#000
    style R fill:#e8f0fe,stroke:#3367d6,stroke-width:2px,color:#000
```

## J. ⭐ Tiered model routing — the cost architecture

```mermaid
flowchart TD
    TASK["Incoming agent task"]
    TASK --> T1["<b>TIER 1 — PLANNING</b> 🧠<br/>GPT-4o / Claude Opus<br/><i>decomposition, complex reasoning</i><br/>~$0.03 per call"]
    T1 --> T2["<b>TIER 2 — EXECUTION</b> ⚙️<br/>Llama 3 8B / GPT-4o-mini<br/><i>tool calls, structured output</i>"]
    T2 --> T3["<b>TIER 3 — GUARDS &amp; ROUTING</b> 🛡️<br/>Phi-3 / TinyLlama<br/><i>classification, filtering</i><br/>~$0.0001 per call"]

    ECON["💰 <b>The economics</b><br/><b>300× cost gap</b> between planning ($0.03)<br/>and classification ($0.0001)<br/><br/>Tiered routing: <b>3.7× cheaper</b><br/>At 100K tasks/day → saves <b>$17,600/day</b> = <b>$6.4M/year</b><br/><br/>⚡ Also <b>2.6× faster</b> end-to-end (6.4s → 2.5s)<br/><br/>🛡️ <b>Guard agents</b> pre-filtering before the LLM<br/>save <b>$50K+/year</b> on their own"]

    SWEET["🎯 <b>SLM sweet spots</b><br/>Phi-3 → routing<br/>Llama 3 8B → tool calls<br/>TinyLlama → guards<br/><br/>📱 <b>On-device:</b> TinyLlama / Gemma enable<br/>privacy-first, offline, zero-cost edge agents"]

    T3 -.-> ECON
    T3 -.-> SWEET

    style T1 fill:#fce8e6,stroke:#c5221f,stroke-width:2px,color:#000
    style T2 fill:#fff3e0,stroke:#e8710a,stroke-width:2px,color:#000
    style T3 fill:#e6f4ea,stroke:#137333,stroke-width:2px,color:#000
    style ECON fill:#fef7e0,stroke:#f9ab00,stroke-width:3px,color:#000
```

## K. ⭐ Master correlation — M5 → later modules

```mermaid
flowchart LR
    subgraph M5C["MODULE 5 CONCEPT"]
        direction TB
        a["ReAct loop<br/><i>5.2.1</i>"]
        b["Tool schemas ·<br/>function calling<br/><i>5.2.3</i>"]
        c["Agent memory ·<br/>vector DB<br/><i>5.1.4</i>"]
        d["LangGraph state graphs<br/><i>5.3.6</i>"]
        e["Tiered model routing<br/><i>5.1.7</i>"]
        f["Retry budgets ·<br/>loop guards<br/><i>5.2.6</i>"]
        g["Plan repair<br/><i>5.3.3</i>"]
        h["Agent trace logging<br/><i>5.2.2</i>"]
        i["RAG pipeline<br/><i>5.4</i>"]
    end

    subgraph LATER["BECOMES"]
        direction TB
        A["<b>M9</b> Production agent orchestration"]
        B["<b>M9</b> MCP &amp; A2A protocols"]
        C["<b>M9</b> RAG evaluation (RAGAS)"]
        D["<b>M9</b> Multi-agent coordination"]
        E["<b>M9</b> Cost tracking &amp; serving"]
        F["<b>M8</b> Guardrails &amp; safety"]
        G["<b>M9</b> Failure recovery"]
        H["<b>M9</b> Observability (Langfuse)"]
        I["<b>M8</b> Grounding &amp; hallucination"]
    end

    a --> A
    b --> B
    c --> C
    d --> D
    e --> E
    f --> F
    g --> G
    h --> H
    i --> I

    LATER --> CAP(["🎓 <b>M10 CAPSTONE</b><br/>production agentic system"])

    style M5C fill:#e8f0fe,stroke:#3367d6,color:#000
    style LATER fill:#e6f4ea,stroke:#137333,color:#000
    style CAP fill:#f3e8fd,stroke:#8430ce,stroke-width:3px,color:#000
```

---

# 5.1 Agent Fundamentals

## 5.1.1 — What is an AI Agent?

> **Concept ID:** `AIAGFNTH000001` · Bloom **B2** · Prereq: M1–M4

### The three limitations agents solve

| # | Limitation | Concrete failure |
|---|---|---|
| **1** | **Stateless** | *Turn 1:* "What flights BLR→BOM on Mar 15?" → "IndiGo 6:30 AM…" · *Turn 2:* "Book the IndiGo one" → **"I don't know which flight you're referring to."** Each call is independent |
| **2** | **No tools** | Cannot browse the web, query databases, call APIs, or execute code |
| **3** | **Single turn** | Cannot iterate toward a solution |

### The definition

> An **AI Agent** is a software system that uses an **LLM as its reasoning engine** to **perceive inputs, reason about them, act via tools, and observe results** — looping until a goal is met.

### The core loop

$$\textbf{Perceive} \to \textbf{Reason} \to \textbf{Act} \to \textbf{Observe} \;\circlearrowleft\; \text{repeat until goal met}$$

> ### 🔑 **A chatbot responds. An agent solves. The loop is what makes the difference.**

### Five capabilities — remove any one and it degrades to a chatbot

| # | Capability | What it provides |
|---|---|---|
| **1** | **Perception** | Receives observations: user input, API responses, tool outputs, sensor data |
| **2** | **State management** | Maintains information across steps — what's done, what remains, what changed |
| **3** | **Reasoning** | Uses the LLM to interpret observations, plan next steps, decide actions via CoT |
| **4** | **Tool use** | Executes actions: API calls, web search, code execution, DB queries, file operations |
| **5** | **Iteration** | Loops **without waiting for a human prompt at each step** — autonomy |

### Chatbot vs Agent

| Dimension | **Chatbot** | **AI Agent** |
|---|---|---|
| Turns | Single turn | **Multi-step loop** |
| Tools | None | Web, DB, APIs, code |
| Memory | Context only | **Context + persistent store** |
| Autonomy | Responds | **Solves** |

### The agent spectrum

**Level 0** (raw LLMs) → **Level 1** (tool use) → **Level 2** (memory) → **Level 3** (planning) → **Level 4** (multi-agent). Each level adds capability.

**Deployed at scale today:** **Devin** (coding) · **Klarna** (customer support) · **Elicit** (research) · **Claude Code** (terminal).

---

## 5.1.2 — Agent Anatomy: Perception, Decision, Action

Every agent — from a thermostat to Devin — has **four components**:

| Component | Role | Example (GitHub Copilot Workspace) |
|---|---|---|
| **Perception** | Observations in — *eyes and ears* | Issue description + codebase |
| **Memory** | Short-term (context window) + long-term (vector DB) | File AST index |
| **Reasoning** | The LLM interprets observations + memory, decides the next action — *the brain* | LLM plans the change |
| **Action** | Tool calls, file writes, API calls, output generation | Edits files, runs tests |

> **The loop is iterative:** each action produces an observation that **feeds back into perception**.

**Token budget reality:** a full agent resolution costs **< $0.001** — orders of magnitude cheaper than a human agent. *This is the economic argument for agentic systems in one line.*

---

## 5.1.3 — Simple Reactive Agents

> **Definition:** a reactive agent selects actions based **solely on the current percept**. No internal state.

$$\text{action} = \text{rules}(\text{percept})$$

**The rule table is static.** It does not learn, adapt, or remember. **First match wins.**

```python
# Smart thermostat — runs every 30 seconds
if sensor.temp > 25:   turn_on(AC)
elif sensor.temp < 20: turn_on(heater)
else:                  standby()
# No memory of how long AC has been on. No weather forecast.

# Email priority filter
if "urgent" in email.subject:  flag_priority("high")
elif sender in VIP_list:       flag_priority("medium")
# Keyword match only. Can't understand context or detect sarcasm.

# Intent router
if classify(msg) == "refund": route_to(refund_agent)
elif classify(msg) == "track": route_to(tracking_agent)
```

### Five limitations

**No memory** · **no planning** · **brittle** · **no learning** · **no composability**.

### ⭐ But still valuable in production

> **Guard agents, IoT, routing, spam filters — fast and free.**
>
> **Guard agents save $50K+/year** by pre-filtering *before* the LLM is invoked.

**Use reactive when:** fixed categories · single-step · speed-critical · **$0 cost target**.

---

## 5.1.4 — Agents with Memory

### Two memory types

| | **In-context (short-term)** | **External (long-term)** |
|---|---|---|
| Where | The context window | Vector DB / KV store |
| Speed | **Fast** | Slight latency |
| Capacity | **Bounded** | **Scalable** |
| Persistence | **Ephemeral** | **Persistent** |

> ⭐ **External memory is ~2,000× cheaper than stuffing everything in-context at scale.**

**What memory enables:** **personalisation** · **progress tracking** · **cross-session learning**.

**Production stack:** **Pinecone / Weaviate** for vectors + **Redis** for key-value + the **context window** for immediate state.

### ⚠️ Four failure modes to watch

**Stale data** · **retrieval misses** · **context overflow** · **privacy leakage**.

---

## 5.1.5 — Agent Goals and Objectives

### Three goal types

| Type | Source |
|---|---|
| **Explicit** | User-provided |
| **Implicit** | System prompt |
| **Decomposed** | **Agent-generated sub-goals** |

> **Success criteria define when to stop.** Without them, agents **loop forever or quit too early**.

### ⚠️ Goal drift — the #1 failure mode

> **Goal drift is the number one failure mode in long-running agents.** The fix: **re-inject the goal every N steps.**

**Sobering benchmark:** complex decomposed goals achieve only **~65% completion** — *the hardest frontier in agentic AI.*

**Task-completion vs open-ended** service require **different goal architectures** — one-off jobs vs continuous operation.

---

## 5.1.6 — LLMs as Agent Brains

### Four capabilities that make LLMs work as controllers

1. **Natural language understanding**
2. **Structured tool calls**
3. **Chain-of-thought reasoning**
4. **Context management**

> ### ⚠️ The framing that matters
> **LLMs are controllers, not thinkers** — remarkably effective **pattern matchers, not comprehenders**.

**Tool calling is native** in GPT-4, Claude, Llama 3.1+, Mistral, Gemini — **no fine-tuning needed**.

**Model choice matters:** a **300× cost gap** between planning ($0.03) and classification ($0.0001).

**Real limitations to design around:** **hallucination** · **context limits** · **no learning between runs**.

---

## 5.1.7 — SLMs as Lightweight Agent Engines

> **The industry trend: big brain for planning, small brains for execution.**

### Tiered routing

| Tier | Model | Job |
|---|---|---|
| **1** | GPT-4o / Claude Opus | **Planning**, decomposition, complex reasoning |
| **2** | Llama 3 8B / GPT-4o-mini | **Execution** — tool calls, structured output |
| **3** | Phi-3 / TinyLlama | **Guards and routing** — classification, filtering |

### The numbers

| Metric | Impact |
|---|---|
| **Cost** | **3.7× cheaper**. At 100K tasks/day: **$17,600/day saved = $6.4M/year** |
| **Latency** | **2.6× faster** end-to-end (6.4s → 2.5s) |

**SLM sweet spots:** **Phi-3** for routing · **Llama 3 8B** for tool calls · **TinyLlama** for guards.

**On-device agents:** TinyLlama / Gemma enable **privacy-first, offline, zero-cost** edge deployment.

---

## 5.1.8 — Building a Simple Agent (coding lab)

**Three components to code:**

| Component | LangChain class | Maps to theory |
|---|---|---|
| **Tools** | `Tool` | **Action** |
| **Brain** | `ChatOpenAI` | **Reasoning** |
| **Loop** | `create_react_agent` | **Iteration** |

> **Theory → code mapping is direct:** Perception = tool results · Reasoning = LLM · Action = tool calls.

⭐ **The agent trace is your debugging tool** — every Thought/Action/Observation is logged and inspectable.

**Cost per query: ~$0.0004** for a multi-step reasoning task with GPT-4o-mini.

> **This coding pattern is the foundation for everything in 5.2–5.7 and Module 9.**

---

# 5.2 The ReAct Framework

## 5.2.1 — ReAct: Reasoning + Acting

> **ReAct interleaves Chain-of-Thought reasoning with tool actions in one context window.** It is **the dominant agent architecture**.

### Why interleave?

| Alone | Problem |
|---|---|
| **Reasoning-only** (pure CoT) | **Hallucinates** — no grounding in real data |
| **Acting-only** (pure tool use) | **Opaque** — no visible rationale, no error diagnosis |
| **ReAct** | ✅ **Solves both** |

**Yao et al. (2022) proved it:** **+6% accuracy on HotpotQA**, **+26% on ALFWorld**.

**Every major framework uses ReAct:** LangChain · OpenAI Assistants · Claude · LangGraph.

**Three failure modes:** hallucinated tools · infinite loops · wrong tool selection *(see Diagram E)*.

---

## 5.2.2 — The Thought-Action-Observation Loop

### Four steps per iteration

| Step | What |
|---|---|
| **Thought** | Reason aloud — model-generated |
| **Action** | A structured tool call |
| **Observation** | The result — ⭐ **the only real-world data in the trace** |
| **Repeat or Answer** | Decide whether to continue |

> ### 🔑 The symmetry
> **Thoughts make actions intentional; Observations make thoughts grounded.** Both are logged.

### Trace economics

- **3-iteration trace ≈ 920 tokens ≈ $0.0006** at GPT-4o-mini
- **Budget ~200 tokens per iteration**
- ⚠️ **Don't let iterations consume >50% of the context window**
- **Observations are ~25% of tokens** — truncate long tool results

### ⭐ Debug backwards

> **Start from the wrong answer and trace back through Observations to find the root cause.** Forward debugging wastes time; the error almost always entered at an Observation or a tool selection.

---

## 5.2.3 — Tool Use and Function Calling

> **Tools transform LLMs from text generators into agents that can act in the real world.**

### The four-step pipeline

```
① Define schemas → ② LLM selects → ③ Orchestrator executes → ④ Result injected back into context
```

> ⚠️ **Note step ③: the *orchestrator* executes, not the LLM.** The model only emits a structured request. This separation is the entire security boundary of an agent system.

### ⭐ Description quality determines selection accuracy

> Write tool descriptions like prompts. **Be specific and include negative examples** ("do NOT use this for…").

### Always validate before execution

Check: **tool name** (exists in registry) · **parameter types** · **bounds** · **permissions**.

**All major LLMs support tool calling natively** — GPT-4, Claude, Llama 3.1+, Mistral, Gemini.

---

## 5.2.4 — ReAct Loop Visualisation

- **Visualisation makes agent behaviour tangible** — you see the branching, looping, and decision points
- **Observations (blue) are the only real-world data** — everything else is model-generated
- **Debug backwards:** Answer → Observation → Action → Thought
- **Context token distribution matters:** Observations ≈ 25% of tokens

---

## 5.2.5 — When to Reason vs. Act

> **The meta-cognitive question:** after every Observation, does the agent have enough to answer, or does it need to think/act more?

### Getting it wrong is costly in both directions

| Failure | Cost |
|---|---|
| **Thought loop** — thinks and thinks, never acts | Burns tokens, produces nothing |
| **Action storm** — fires tools without planning | Random searches, irrelevant results, burns API budget |

### Reasoning depth heuristics

| Task complexity | Thought depth |
|---|---|
| Simple lookup | **Minimal** (1 sentence) |
| Analysis | Moderate |
| Multi-constraint research | **Deep** |

> ### 📏 The human heuristic
> **If you'd spend <3 seconds deciding as a human → minimal Thought. >30 seconds → deep Thought.**

### Three design patterns

**Think-Once-Act-Once** · **Interleaved ReAct** · **Plan-Then-Execute**

### Calibration by task tier

| Tier | Iterations |
|---|---|
| Lookup | **2** |
| Analysis | **5–7** |
| Research | **10–15** |

**Enforce in the prompt:** *"Act within 2 Thoughts"* and *"max N tool calls."*

> 🔑 **The reason-vs-act balance is a design decision, not an emergent property. Engineer it deliberately.**

---

## 5.2.6 — Self-Reflection and Iterative Refinement

> **Basic ReAct doesn't self-check.** The **Reflexion** pattern adds: **generate → evaluate → reflect → retry.**

**Shinn et al. (2023): +24% on code generation** with *verbal* self-reflection.

### Three evaluation strategies

| Strategy | Cost | When |
|---|---|---|
| **Constraint validation** | **$0** | Does the JSON parse? Do tests pass? |
| **LLM-as-judge** | ~$0.001 | Open-ended quality |
| **Ground-truth comparison** | — | When references exist |

> ### ⚠️ Always set retry budgets (max 3)
> **Without them, reflection loops forever — the #1 deployment risk of this pattern.**

**Use Reflexion for high-stakes tasks where correctness matters more than speed.**

---

## 5.2.7 — Implementing ReAct Agents (coding lab)

Same three components as 5.1.8 — `Tool`, `ChatOpenAI`, `create_react_agent` — now with search and calculator tools.

**Cost per query ~$0.0004.** This pattern is the foundation for Sub-Modules 5.3–5.7 and Module 9.

---

# 5.3 Planning and Decomposition

## 5.3.1 — Classical Planning

> **Classical planning = state-space search with STRIPS actions** (preconditions + effects).

| Property | Detail |
|---|---|
| **Completeness** | If a plan exists, the planner **finds it** |
| **Soundness** | The plan **will work** |
| **Forward chaining** | Search from the start state |
| **Backward chaining** | Search from the goal |

> ⚠️ **Fails on open worlds and natural language** — real tasks break the **closed-world assumption**.

**Still thriving in:** robotics · games · logistics — anywhere the world is **fully specified**.

---

## 5.3.2 — LLM-Based Planning

> LLMs generate plans **from natural language goals — no formal logic needed**.

### Two strategies

| | **Plan-then-execute** | **Interleaved** |
|---|---|---|
| When the plan is made | Full plan up front | One step at a time |
| Adaptability | Lower | Higher |
| Predictability | Higher | Lower |

**CoT planning is the simplest technique** — *"think step by step"* with **structured output**.

### ⚠️ The characteristic failure

> **LLM plans can be plausible but infeasible:** ordering errors, budget violations, missing constraints.

**The fix:** **verification prompts catch many errors** — one extra LLM call before execution saves wasted work.

---

## 5.3.3 — Hybrid Planning Approaches

> **Combining classical structure with LLM flexibility.**

### Three hybrid patterns

1. **LLM → verifier** — LLM plans, a checker validates
2. **Classical skeleton → LLM details** — formal structure, natural-language filling
3. **Iterative plan repair** ⭐

> ### 🔑 Plan repair is the key differentiator
> **Adapt the plan when reality doesn't match expectations** — preserving completed work and replanning only the remaining steps from the current state.

**Economics:** verification costs **~$0.02 extra** but **reduces execution failures by 60%**.

### Match strategy to task

| Task | Approach |
|---|---|
| Simple | Pure LLM |
| Constrained | Hybrid |
| **Safety-critical** | **Full verification** |

**Devin exemplifies hybrid planning:** LLM plans code changes → test suite verifies → replan on failure.

---

## 5.3.4 — Task Decomposition

> **Decomposition = breaking complex goals into atomic sub-tasks** (one tool call each).

### Three types

| Type | Structure |
|---|---|
| **Recursive** | Tree |
| **Parallel** | Independent branches |
| **Sequential** | Dependent chain |

**DAG structure tracks dependencies** — determines what can parallelise and identifies the **critical path**.

> ### 🎯 The sweet spot: **5–8 sub-tasks**
> Too shallow fails to simplify; too deep adds overhead.

**LLM decomposition prompts work well** when you specify **output format + an atomicity rule**.

---

## 5.3.5 — Goal-Directed Behaviour

### Three components

**Persistence** · **progress tracking** · **adaptation**.

### What causes goal drift

**Context overflow** · **attention decay** · **tangential observations** · **vague goals**.

### The prevention toolkit

| Tool | Detail |
|---|---|
| **Re-inject goals** | Restate the objective every N steps |
| **Scope boundaries** | Explicit out-of-scope statements |
| **Step budgets** | Hard iteration caps |
| **Progress checklists** | Explicit state of what's done |

> **Alignment checks every 5 steps cost ~40 tokens but save 5–10 wasted steps.** One of the best ROI decisions in agent design.

**Track progress explicitly** — a context checklist for short tasks, an **external store** for long ones.

---

## 5.3.6 — Cognitive Architectures: Loops to State Graphs

> ### 🔑 **Agent control flow is a designable artifact** — not just a fixed loop, but a **graph you architect**.

**The evolution:** reactive → ReAct → state machine → **state graph**. Each adds control capability.

| Framework | Model | Fit |
|---|---|---|
| **LangGraph** ⭐ | Explicit **state graph**: nodes, **conditional edges**, **cycles**, **persistence** | Branching, checkpointing, durable agents |
| **AutoGen** | **Conversation patterns** | Multi-agent dialogue |
| **Swarm** | **Lightweight handoff** | Simple agent-to-agent routing |

> **Start with ReAct. Upgrade to LangGraph when you need branching, persistence, or checkpointing.**

---

## 5.3.7 — Planning System Implementation (coding lab)

**Build a plan-and-execute agent:**

- **Planner (GPT-4o) + Executor (mini)** — ⭐ tiered routing saves cost while maintaining plan quality
- **Structured planning prompts** with **JSON output + atomicity rules** produce reliable plans
- **Execution loop:** iterate steps → check dependencies → execute with ReAct → handle failures
- ⭐ **Plan repair preserves completed work** — only replans remaining steps from the current state

> **This pattern is the foundation for all production planning agents in Module 9 (LLMOps).**

---

# 5.4 — RAG: Retrieval-Augmented Generation

> Covered in the mentor session (`RAG_Content_covered_by_Mentor.zip`) and two lab notebooks (with / without OpenAI API).

### Why RAG belongs in the agent module

An agent's **memory** (5.1.4) and an agent's **knowledge grounding** are the same mechanism: **retrieve relevant context, inject it into the prompt.** RAG is the retrieval half of agent memory.

### The pipeline

```
Documents → Chunk → Embed → Store in vector DB
                                    ↓
User query → Embed → Similarity search → Top-k chunks → Inject into prompt → LLM → Grounded answer
```

### What RAG changes — and what it doesn't

> ⭐ **RAG does not change the model's weights. It changes the context that drives prediction.** *(Established back in M1 §1.2.2: without retrieval $P(\text{"1969"}) \approx 0.6$; with a NASA document in context, $\approx 0.97$.)*

**What it solves:** knowledge cutoff · hallucination · proprietary/private data · citability.

### Design decisions that matter

| Decision | Consequence |
|---|---|
| **Chunk size** | Too small loses context; too large dilutes relevance |
| **Chunk overlap** | Prevents splitting a fact across a boundary |
| **Embedding model** | ⚠️ Must match between indexing and query time |
| **Top-k** | More chunks = better recall, more tokens, more cost |
| **Chunk ordering** | ⚠️ **Lost-in-the-middle** (M2 §2.2.4) — put the most important chunks **first or last** |

**Stack:** LangChain / LlamaIndex for orchestration · **FAISS, Pinecone, Weaviate, Qdrant, ChromaDB** for vectors.

**Evaluation:** **RAGAS** — faithfulness · answer relevance · context precision · context recall (M2 §2.3.10). → *deepened in Module 9*

---

## Assignments & application

| Item | Detail |
|---|---|
| **`M5_AST_01_Leveraging_LLMs_for_Querying_Insights.ipynb`** | Assignment 1 |
| **`M5_AST_02_Building_Agents_using_LangGraph_(With/Without_OpenAI_API).ipynb`** | ⭐ Assignment 2 — **LangGraph**, ties directly to 5.3.6 |
| `M5_Additional_NB_01_Tool_Calling.ipynb` | Pairs with 5.2.3 |
| `M5_Additional_NB_02_Building_Agent_using_LangChain.ipynb` | Pairs with 5.1.8 / 5.2.7 |
| `M5_Additional_NB_01/02_Building_RAG_Pipeline_using_LangChain_*.ipynb` | Pairs with 5.4 |
| **Stock Prices Insights Generation Application** | `Steps_for_...pdf` + `Application.zip` — end-to-end build |
| `Session_Objectives.pdf` | Tutorial session, 18 Apr 2026 |

> 💡 **Do the "Without OpenAI API" variants.** Running agents on Groq/Ollama forces you to confront tool-calling format differences — which is exactly the lesson of 5.2.3.

---

## Master list of misconceptions

| ❌ Myth | ✅ Reality |
|---|---|
| "An agent is just a chatbot with tools" | ⭐ **A chatbot responds; an agent solves.** The **loop** — autonomous iteration — is the difference |
| "Agents need a special model" | **Tool calling is native** in GPT-4, Claude, Llama 3.1+, Mistral, Gemini. **No fine-tuning needed** |
| "The LLM executes the tool" | ⚠️ **No** — the LLM emits a structured *request*; the **orchestrator executes**. This separation is the security boundary |
| "Reactive agents are obsolete" | **Guard agents save $50K+/year** by pre-filtering before the LLM. Fast, free, still deployed |
| "Put everything in the context window" | **External memory is ~2,000× cheaper at scale** |
| "LLMs think" | ⭐ **LLMs are controllers, not thinkers** — effective pattern matchers, not comprehenders |
| "Use the biggest model for everything" | **300× cost gap** between planning and classification. **Tiered routing is 3.7× cheaper and 2.6× faster** |
| "More reasoning is always better" | **Thought loops** burn tokens and produce nothing. Calibrate by task tier |
| "More tool calls means a more thorough agent" | **Action storms** burn API budget on irrelevant results |
| "ReAct agents self-correct" | ⚠️ **Basic ReAct does not self-check.** You must add **Reflexion** explicitly |
| "Reflexion is strictly better" | Without a **retry budget (max 3)** it loops forever — **the #1 deployment risk** |
| "Agents will stay on task" | ⚠️ **Goal drift is the #1 failure mode** in long-running agents. Re-inject goals every N steps |
| "Agents complete complex goals reliably" | Complex **decomposed goals: ~65% completion**. This is the frontier, not a solved problem |
| "Classical planning is obsolete" | **Completeness and soundness are guaranteed** — still dominant in robotics, games, logistics |
| "An LLM plan that reads well will work" | ⚠️ Plans are often **plausible but infeasible**: ordering errors, budget violations, missing constraints |
| "Verification is an unnecessary cost" | **~$0.02 extra reduces execution failures by 60%** |
| "Deeper decomposition is better" | **Sweet spot is 5–8 sub-tasks.** Too deep adds overhead |
| "When a plan fails, start over" | ⭐ **Plan repair preserves completed work** — replan only the remainder |
| "Use LangGraph from the start" | **Start with ReAct.** Upgrade when you need branching, persistence, or checkpointing |
| "RAG changes what the model knows" | ⚠️ **RAG does not change weights — it changes the context** that drives prediction |
| "Chunk order in RAG doesn't matter" | **Lost-in-the-middle** — put critical chunks **first or last** |

---

## Glossary

| Term | Definition |
|---|---|
| **Action storm** | Failure mode: firing tools without planning |
| **Agent** | System using an LLM as reasoning engine in a perceive→reason→act→observe loop |
| **AutoGen** | Multi-agent framework based on conversation patterns |
| **Backward chaining** | Classical planning search from the goal |
| **Cognitive architecture** | The designed control flow of an agent |
| **DAG** | Directed acyclic graph — tracks sub-task dependencies |
| **Goal drift** | Agent losing focus on the original objective — #1 long-run failure |
| **Guard agent** | Cheap reactive filter running before the LLM |
| **Hallucinated tool** | Agent calling a tool that doesn't exist |
| **LangGraph** | State-graph agent framework with cycles and persistence |
| **Observation** | The tool result — the only real-world data in a ReAct trace |
| **Plan repair** | Replanning only the remaining steps, preserving completed work |
| **Plan-then-execute** | Generate the full plan before acting |
| **ReAct** | Reasoning + Acting — interleaved CoT and tool use (Yao et al. 2022) |
| **Reactive agent** | `action = rules(percept)`; no state |
| **Reflexion** | generate → evaluate → reflect → retry (Shinn et al. 2023) |
| **Retry budget** | Hard cap on reflection retries (max 3) |
| **SLM** | Small Language Model used for execution/guard tiers |
| **STRIPS** | Classical planning formalism: preconditions + effects |
| **Swarm** | Lightweight agent handoff framework |
| **TAO loop** | Thought-Action-Observation |
| **Thought loop** | Failure mode: reasoning without ever acting |
| **Tiered routing** | Matching model size to task tier for cost/latency |
| **Tool schema** | Structured definition (name, description, parameters) the LLM selects from |

---

## References and further study

### 📕 Books

| Book | For Module 5 |
|---|---|
| ⭐ **AI Engineering** — Chip Huyen, O'Reilly 2025 | **The course's designated M5 reference** — agents, RAG, MCP, agentic patterns |
| **Building LLMs for Production** — Bouchard & Peters | LangChain, LlamaIndex, agents, advanced RAG |
| **Hands-On Large Language Models** — Alammar & Grootendorst | RAG and semantic search chapters |

### 📄 Papers — the M5 canon

| Paper | Link | Section |
|---|---|---|
| ⭐ **ReAct: Synergizing Reasoning and Acting** — Yao 2022 | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) | **5.2.1 — required reading** |
| ⭐ **Reflexion** — Shinn 2023 | [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) | 5.2.6 |
| **Toolformer** — Schick 2023 | [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) | 5.2.3 |
| **Tree of Thoughts** — Yao 2023 | [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) | 5.3 |
| **Plan-and-Solve Prompting** — Wang 2023 | [arXiv:2305.04091](https://arxiv.org/abs/2305.04091) | 5.3.2 |
| **RAG** — Lewis 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | 5.4 |
| **Lost in the Middle** — Liu 2023 | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) | 5.4 chunk ordering |
| **Generative Agents** — Park 2023 | [arXiv:2304.03442](https://arxiv.org/abs/2304.03442) | 5.1.4 memory architecture |

### 🔗 Docs & guides

| Resource | Link | For |
|---|---|---|
| ⭐ **Building Effective Agents** — Anthropic | [anthropic.com/research](https://www.anthropic.com/research/building-effective-agents) | **Required reading alongside M5** |
| ⭐ **Model Context Protocol (MCP)** | [modelcontextprotocol.io](https://modelcontextprotocol.io/) | Agent–tool interoperability → M9 |
| **LangGraph docs** | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | ⭐ 5.3.6, Assignment 2 |
| **LangChain Conceptual Guides** | [python.langchain.com/docs/concepts](https://python.langchain.com/docs/concepts) | 5.1.8, 5.2.7 |
| **LlamaIndex docs** | [docs.llamaindex.ai](https://docs.llamaindex.ai/) | 5.4 RAG |
| **OpenAI Guide to Building Agents** | [platform.openai.com/docs/guides/agents](https://platform.openai.com/docs/guides/agents) | 5.1–5.2 |
| **Berkeley LLM Agents MOOC** | 11 lectures, Berkeley RDI | ⭐ **Directly supports M5** |
| **AutoGen** | [microsoft.github.io/autogen](https://microsoft.github.io/autogen/) | 5.3.6 |

### 📌 Study strategy for Weeks 9–10

1. **Read the ReAct paper first** — it is short and the entire module is downstream of it
2. Build the **5.1.8 simple agent** before reading 5.2. Watching a real trace print makes the TAO loop obvious
3. **Deliberately break your agent:** give it a tool with a vague description, then a good one. Observe selection accuracy change. That is 5.2.3 in one experiment
4. **Add a max-iteration cap and watch it save you** — you will hit an infinite loop within the first hour
5. Do **Assignment 2 (LangGraph)** only after the LangChain agent works. The upgrade path in 5.3.6 is real advice
6. Read **Anthropic's "Building Effective Agents"** — it is the best short industry counterweight to framework marketing

---

## Self-check question bank

### 5.1 Fundamentals
1. Name the three limitations of standalone LLMs that agents solve.
2. Write the core agent loop.
3. State the one-sentence distinction between a chatbot and an agent.
4. Name the five capabilities that define an agent. What happens if you remove one?
5. Name the four anatomy components and give a real system that maps to each.
6. Write the reactive-agent equation. Name three production uses where it is still correct.
7. Name the five limitations of reactive agents.
8. Compare in-context vs external memory on speed, capacity, persistence, and cost.
9. Name the four memory failure modes.
10. Name the three goal types.
11. What is goal drift, and what is the standard mitigation?
12. What completion rate do complex decomposed goals achieve?
13. Why are LLMs described as "controllers, not thinkers"?
14. What is the cost gap between planning and classification calls?
15. Describe the three routing tiers and name a model for each.
16. What does tiered routing save at 100K tasks/day?

### 5.2 ReAct
17. What does reasoning-only fail at? Acting-only? How does ReAct fix both?
18. What gains did Yao et al. (2022) report, on which benchmarks?
19. Name the four steps of one ReAct iteration.
20. Which element of a trace is the only real-world data?
21. Roughly how many tokens and dollars is a 3-iteration trace?
22. In which direction should you debug an agent trace, and why?
23. Name the three ReAct failure modes and one fix each.
24. Write the four-step tool-calling pipeline. Who executes the tool?
25. What four things should you validate before executing a tool call?
26. What determines tool selection accuracy?
27. Contrast the thought loop and the action storm.
28. Give the human heuristic for reasoning depth.
29. How many iterations for a lookup? An analysis? A research task?
30. Name the three reason/act design patterns.
31. Write the Reflexion cycle. What gain did Shinn et al. report?
32. Name the three evaluation strategies and their costs.
33. What is the #1 deployment risk of Reflexion, and the fix?

### 5.3 Planning
34. What two guarantees does classical planning provide?
35. What assumption does classical planning make that breaks on real tasks?
36. Contrast forward and backward chaining.
37. Contrast plan-then-execute with interleaved planning.
38. What is the characteristic failure of LLM-generated plans?
39. Name the three hybrid planning patterns.
40. What does verification cost, and what does it buy?
41. What is plan repair, and why does it matter more than replanning from scratch?
42. Name the three decomposition types. What is the sweet-spot sub-task count?
43. What does a DAG give you in decomposition?
44. Name four causes of goal drift and four prevention tools.
45. What does an alignment check cost and save?
46. Trace the cognitive architecture evolution in four steps.
47. When should you upgrade from ReAct to LangGraph?
48. Compare LangGraph, AutoGen, and Swarm.

### 5.4 RAG
49. Does RAG change model weights? What does it change?
50. Name five RAG design decisions and their consequences.
51. Why does chunk ordering matter?
52. Name the four RAGAS metrics.

---

*Study notes compiled from the Module 5 source decks. Concept IDs preserved for cross-referencing.*
*Series: [M1](../M1/M1_Study_Notes.md) · [M2](../M2/M2_Study_Notes.md) · [M3](../M3/M3_Study_Notes.md) · [M4](../M4/M4_Study_Notes.md) · **M5** · M6 · M7 · M8 · M9 · M10*
