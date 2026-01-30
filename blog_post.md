# The Physics of Failure: Why Naive RAG Breaks

Naive RAG works fine for simple demos.
It starts breaking the moment you introduce real complexity.

The reason isn’t bugs or bad tuning — it’s structural. These are **failure modes baked into the Retrieve → Read design**.

Here are the big ones, demonstrated with a T20 World Cup 2026 dataset.

---

### 1. “Lost in the Middle”

LLMs don’t treat all context equally.

They pay the most attention to:

* the **beginning** of the prompt (primacy)
* the **end** of the prompt (recency)

Everything in the middle? Much weaker signal.

In Naive RAG, we usually:

* retrieve Top-K chunks
* concatenate them
* hope the model finds the right sentence

If the real answer lives in chunk #4 of 10 — say, a small detail about the **Oman men’s cricket team’s confidence** buried between squad lists — chances are the model simply ignores it.

The system *retrieved* the answer.
The generation step just never used it.

---

### 2. The Multi-Hop Reasoning Gap

Naive RAG is good at **single-hop** questions:

> “Which team replaced Bangladesh in the tournament?”
> **Answer:** Scotland.

It falls apart on **multi-hop** questions:

> “Which group is the team that replaced Bangladesh in?”

If:
* Bangladesh → replaced by **Scotland** (Chunk A)
* **Scotland** → added to **Group C** (Chunk B)

Naive RAG retrieves chunks mentioning *Bangladesh* — but often misses *Group C*, because the query doesn't mention "Scotland", the actual bridge.

There’s no loop that says:
> “I found that Scotland replaced Bangladesh. Now, what group is Scotland in?”

In our system, it simply returned **"Scotland"** again, completely missing the group information.

Naive RAG has retrieval.
It does **not** have reasoning-guided retrieval.

---

### 3. Extraction Errors Under Load

Even when the right text is present, Naive RAG often extracts the **wrong entity** or an incomplete one.

Example query:
> “Who is hosting the ICC Men’s T20 World Cup 2026 and when does it start?”

The text clearly says: *"The tournament will be co-hosted by India and Sri Lanka... begin on February 7, 2026."*

Our Naive RAG system answered:
> **"February 7, 2026"**

It saw the date and stopped. It completely missed the "India and Sri Lanka" part because handling compound questions requires multi-step extraction logic that simple vector search + generation often flubs without chain-of-thought.

Having the answer in context is not the same as *using it correctly*.

---

### 4. Context Window Collapse

When retrieval feels weak, people increase **K**.

That usually makes things worse.

More chunks → more noise → diluted signal → blown context window.

If we blindly added every squad update, injury report, and match schedule into the context, the LLM would likely hallucinate relationships between unrelated players.

Naive RAG has no compression, refinement, or prioritization step.
It just keeps stuffing.

---

### 5. Embedding Drift (Silent Failure)

This one is subtle — and dangerous.

If:
* your embedding model gets updated
* old documents (like the 2024 World Cup rules) aren’t re-indexed

You now have **fractured vector space**.

Queries still run.
No errors get thrown.
But retrieval quality quietly collapses.

Naive RAG pipelines rarely check for this mismatch.
The system looks “healthy” while returning garbage.

---

### 6. Retrieval Timing Issues

In async systems, retrieval and generation aren’t always perfectly coordinated.

If generation fires:
* before retrieval completes
* or with stale cached results (e.g., using an old captain's name like Alyssa Healy instead of Sophie Molineux)

The LLM responds with little or no context.

Instant hallucination.

Naive RAG assumes retrieval always finishes cleanly before generation.
Real systems don’t behave that politely.

---

## The takeaway

Naive RAG doesn’t fail because LLMs are bad.
It fails because **retrieval, reasoning, and context management are treated as a single step**.

They aren’t.

In the next post, I’ll break down how modern RAG systems try to fix these issues — and why many of those fixes still fall short.
