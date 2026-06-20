# Lab 2 — Results

Retrieval verified live with local embeddings (`EMBED_BACKEND=local`). The final
**generation** step needs a Gemini key — set `GOOGLE_API_KEY` and re-run to get
the cited answers / the "I don't know" decline.

| Question | Top retrieved | Expected answer behaviour |
|---|---|---|
| How long do I have to get a full refund? | **kb-04** (policy.md) | "Full refund within 30 days… (source: policy.md)" |
| How do I reset my password? | **kb-07** (it.md) | "Login screen → 'Forgot password'… (source: it.md)" |
| What is the company's stock price today? | only irrelevant passages (kb-10/09/06) | **decline** — "I don't know — the knowledge base doesn't cover this." |

The out-of-scope question retrieves nothing relevant, so the grounding rule in
`build_prompt` forces a decline instead of an invented stock price.

**Stretch (k=1 vs k=3):** too little context (k=1) risks missing the passage and
forcing a false "I don't know"; too much buries the answer in noise and invites
wrong citations. k=3 is the sweet spot for this small KB.

Indexing (`build_index`) and querying (`retrieve`/`answer`) are separate functions.
