# Speaker Script & Q&A Prep

Converted from the team's Word document. Covers slides 4–6 plus prepared Q&A answers.

\[02:30 - 04:00\] Slide 4: Key Decisions & Trade-offs (40% Weight)

(Advance to Slide 4)

"Designing for financial research required careful technical trade-offs:

1\. Structure-Aware vs. Fixed-Size Chunking: We chose structure-aware
chunking to preserve table boundaries and financial headers, trading
away uniform processing speed for semantic integrity.

2\. RAG vs. Model Fine-Tuning: We chose RAG to guarantee real-time
citation accuracy and eliminate the high hallucination risks and massive
compute costs of fine-tuning.

3\. Local MiniLM vs. Paid API Embeddings: We selected local MiniLM
embeddings for zero cost, zero latency, and persistent offline stability
for our live demo, accepting a minor trade-off in semantic depth.

4\. Strict Refusal vs. LLM Guessing: In finance, guessing creates
liability. We traded away 'helpful-sounding' speculative answers to
guarantee 100% desk trust."

\[04:00 - 05:15\] Slide 5: What Broke & Tough Documents (15% Weight)

(Advance to Slide 5)

In our corpus audit, the system successfully extracted 4 email
attachments, blocked 1 restricted record, and verified 0 true
duplicates.

"Auditing real-world files revealed three major edge cases:

Security Injections: In the Data Quality Memo, we discovered malicious
prompt injections attempting to artificially skew KRNX dividend outputs.
We built an isolation audit layer to detect and neutralize injected
prompts before reaching the LLM.

Hidden Spreadsheets: An Excel rates model contained a restricted
Internal_Overlay tab. We upgraded our parser to inspect sheet metadata
and strictly exclude hidden sheets.

Trapped Visual Data: Point forecasts were locked inside PPT image
charts. Rather than hallucinating guesses, our system gracefully refuses
to answer queries dependent on unseen visual data."

\[05:15 - 05:45\] Slide 6: Path to Production & Q&A

(Advance to Slide 6)

"Our POC achieved 100% success on security refusals and citations,
alongside a 50% hit rate on deep cross-doc queries.

To scale for production, we will:

Implement Hybrid Search (BM25 + Vector) with a Cross-Encoder Reranker.

Add Multimodal OCR for visual PPT charts.

Integrate RBAC with Active Directory for permissions.

Automate testing using RAGAS metrics.

Q1: Why did you choose RAG over Fine-Tuning the LLM?

Answer: Fine-tuning adapts model style and tone rather than storing
reliable, real-time facts, and it carries a high risk of hallucination.
For an investment desk receiving 54 unstructured files daily,
fine-tuning cannot provide real-time updates or granular document-level
citations. RAG guarantees 100% source traceability and allows us to
enforce strict zero-hallucination refusals.

Q2: Why choose local all-MiniLM-L6-v2 embeddings instead of paid API
embeddings (e.g., OpenAI or Gemini)?

Answer: Local MiniLM embeddings provide zero cost, zero API latency, and
persistent offline vector indexing. This eliminates external network
dependencies and rate limits, guaranteeing absolute stability during our
live demo. While we traded away minor semantic depth, local execution
ensured deterministic performance and zero budget consumption.

Q3: Why enforce strict LLM refusal instead of allowing the model to make
helpful guesses?

Answer: In financial research, a plausible-sounding hallucination
creates compliance liability and breaks user trust. Analysts need
absolute confidence that generated outputs are grounded in verifiable
source documents. We deliberately traded away "helpful-sounding"
speculative answers to ensure 100% desk trust.

Handling Tough Documents & Edge Cases (15% Weight Category)

Q4: How did you detect and neutralize the prompt injection in the Data
Quality Memo?

Answer: During our document audit, we identified malicious prompt text
within the Data Quality Memo attempting to override system instructions
and alter KRNX dividend outputs. We built an isolation audit layer prior
to LLM generation that detects, flags, and strips prompt injection
patterns before retrieved context is passed to Gemini.

Q5: How did your system handle the hidden Internal_Overlay sheet in the
rates model?

Answer: Standard text parsers often extract hidden Excel sheets
indiscriminately. We upgraded our parser to inspect sheet visibility
metadata (e.g., checking sheet_state != 'visible'), ensuring restricted
internal sheets are strictly excluded from indexing to prevent
compliance breaches.

Q6: How did you handle point forecasts trapped inside PPT chart images?

Answer: Because our text parser could not extract embedded image text,
our system adheres to strict refusal principles rather than letting the
LLM guess missing visual numbers. We have slated Multimodal/OCR
integration as a primary upgrade for production scaling.

Architecture & Implementation Details

Q7: How does Structure-Aware Chunking work, and why is it superior to
fixed-size chunking?

Answer: Fixed-size chunking arbitrary cuts off financial tables and
section headers mid-sentence, destroying tabular context.
Structure-aware chunking parses document headers and table boundaries
dynamically, producing 282 coherent chunks that preserve tabular and
semantic integrity.

Q8: Why set Top-k retrieval to 3? How will you handle deep
cross-document queries in production?

Answer: Retrieving the top-3 evidence chunks optimized latency and
context relevance for single-document queries in our POC, yielding a
100% success rate on citations. However, deep cross-document queries hit
a 50% hit rate limit. For production, we will implement Hybrid Search
(BM25 + Dense Vectors) paired with a Cross-Encoder Reranker to aggregate
and synthesize information across multiple files.
