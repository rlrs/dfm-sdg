# Tool Use Pack

`tool_use` is a starter pack for structured tool-calling tasks.

The goal is to synthesize interactions where the model must choose a tool, produce valid arguments, consume tool results, and finish with a grounded assistant answer. This deserves its own pack because the row schema, verification, and failure modes differ from plain text synthesis.

## Start here

1. Begin with one narrow tool schema such as calculator, retrieval, or calendar lookup.
2. Lock down the call format before adding multi-step traces.
3. Add validators for argument shape, tool choice, and final answer grounding.
4. Only then expand to repair turns, retries, or deep-research style chains.

## Dolci inspirations

- `olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated_S2`
- `olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated_T2`
- `olmo-toolu-s2-sft-m3`
- `olmo-toolu-s2-sft-m4v2`
- `olmo-toolu-s2-sft-m5v2`
- `olmo-toolu_deepresearch_no_thinking_DRv4_DS`

The scaffold build produces starter rows and a guide artifact. It does not implement real tool execution yet.
