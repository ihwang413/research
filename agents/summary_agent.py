from __future__ import annotations

import json

import anthropic

from utils.data_models import Paper

SUMMARY_SYSTEM_PROMPT = """You are an expert academic literature analyst specializing in systematic literature reviews. Your task is to produce structured summaries of academic paper abstracts.

## Output Format

For each paper, produce a summary using EXACTLY this three-line format:
Purpose: [One concise sentence describing the research goal or question]
Method: [One concise sentence describing the methodology, study design, or approach]
Results: [One concise sentence describing the main findings or outcomes, or "Not reported" if absent]

## Evaluation Rubric

**Purpose line** must:
- State WHAT the paper investigates (the research question or objective)
- Be specific enough to distinguish this paper from related work
- Avoid vague phrases like "this paper studies" — state the actual subject

**Method line** must:
- Name the specific research method (e.g., RCT, meta-analysis, survey, case study, simulation, experiment)
- Include sample size or dataset size when available
- Mention key tools or frameworks if central to the method

**Results line** must:
- State the primary finding with specificity (numbers, effect sizes, comparisons when available)
- If the abstract doesn't report results, write exactly: Not reported
- Avoid generic statements like "results showed improvements"

## Quality Standards

- Each line must be 15-150 characters long
- Total summary must be 50-800 characters
- Do not copy sentences verbatim from the abstract
- Do not include author names or citations
- Do not add extra lines or explanations outside the three-line format

## Worked Examples

**Example 1**
Abstract: "This study examines the effectiveness of AI tutoring systems on student math performance. We conducted a randomized controlled trial with 342 middle school students over 12 weeks, comparing an AI-powered adaptive system against traditional instruction. Students using the AI system showed a 23% improvement in algebra scores compared to 8% for the control group."

Summary:
Purpose: To evaluate whether AI tutoring systems improve middle school students' mathematics performance compared to traditional instruction.
Method: Randomized controlled trial with 342 middle school students over 12 weeks comparing AI adaptive tutoring versus conventional teaching.
Results: AI tutoring group achieved 23% algebra score improvement versus 8% for controls, a statistically significant difference.

**Example 2**
Abstract: "Large language models have transformed natural language processing. We review 87 papers published between 2018-2023 on LLM applications in education, categorizing findings by pedagogical use case, effectiveness evidence, and implementation challenges."

Summary:
Purpose: To systematically map how large language models are applied in educational settings and synthesize evidence on their effectiveness.
Method: Systematic review of 87 papers (2018–2023) categorized by pedagogical use case, effectiveness evidence, and implementation barriers.
Results: Not reported

**Example 3**
Abstract: "Formative assessment practices remain inconsistent across K-12 schools. This mixed-methods study surveyed 1,200 teachers and conducted 40 classroom observations to identify barriers to effective formative assessment implementation."

Summary:
Purpose: To identify barriers preventing consistent implementation of formative assessment practices across K-12 classrooms.
Method: Mixed-methods study combining a survey of 1,200 teachers with 40 structured classroom observations.
Results: Not reported

## Batch Processing Instructions

You will receive a numbered list of papers. Return a JSON array (same order as input) where each element is a string containing the three-line summary in the exact format above.

If a paper's abstract is marked as "Abstract not available", return: "Purpose: \nMethod: \nResults: Abstract not available"

Return ONLY the JSON array — no explanation, no markdown fences, no extra text.
"""

SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": SUMMARY_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


async def run(papers: list[Paper], config: dict) -> list[Paper]:
    api_cfg = config.get("api", {})
    sum_cfg = config.get("summary", {})
    model = sum_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = sum_cfg.get("max_tokens", 512)
    batch_size = sum_cfg.get("batch_size", 10)

    api_key = api_cfg.get("anthropic_key")
    client = anthropic.AsyncAnthropic(api_key=api_key if api_key else None)

    needs_summary = [(i, p) for i, p in enumerate(papers) if p.abstract is not None]
    no_abstract = [(i, p) for i, p in enumerate(papers) if p.abstract is None]

    for i, p in no_abstract:
        papers[i] = p.model_copy(update={"summary": "Purpose: \nMethod: \nResults: Abstract not available"})

    total = len(needs_summary)
    processed = 0

    for batch_start in range(0, total, batch_size):
        batch = needs_summary[batch_start: batch_start + batch_size]
        summaries = await _summarize_batch(client, model, max_tokens, batch)

        for (orig_idx, paper), summary in zip(batch, summaries):
            if not _validate_summary(summary):
                retry = await _summarize_batch(client, model, max_tokens, [(orig_idx, paper)])
                summary = retry[0] if retry else "Purpose: \nMethod: \nResults: Summary generation failed"
            papers[orig_idx] = paper.model_copy(update={"summary": summary})

        processed += len(batch)
        print(f"[summary] {processed}/{total} papers summarized", end="\r", flush=True)

    print(f"\n[summary] Done.", flush=True)
    return papers


async def _summarize_batch(
    client: anthropic.AsyncAnthropic,
    model: str,
    max_tokens: int,
    indexed_papers: list[tuple[int, Paper]],
) -> list[str]:
    lines = []
    for n, (_, paper) in enumerate(indexed_papers, 1):
        abstract = paper.abstract or "Abstract not available"
        lines.append(f"{n}. Title: {paper.title}\nAbstract: {abstract}")
    user_text = "\n\n".join(lines)

    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens * len(indexed_papers),
            system=SYSTEM_BLOCKS,
            messages=[{"role": "user", "content": user_text}],
        )
        usage = resp.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_write = getattr(usage, "cache_creation_input_tokens", 0)
        if cache_read or cache_write:
            print(f"\n[summary] cache hit={cache_read} write={cache_write}", flush=True)

        raw = resp.content[0].text.strip()
        # Claude sometimes wraps JSON in markdown code fences — strip them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(indexed_papers):
            return [str(s) for s in parsed]
    except (anthropic.APIError, json.JSONDecodeError, IndexError):
        pass

    return ["Purpose: \nMethod: \nResults: Summary generation failed"] * len(indexed_papers)


def _validate_summary(summary: str) -> bool:
    if not summary:
        return False
    has_purpose = "Purpose:" in summary
    has_method = "Method:" in summary
    has_results = "Results:" in summary
    length_ok = 50 <= len(summary) <= 800
    return has_purpose and has_method and has_results and length_ok
