# CS399 mediator prototype

The sender submits the same query in both runs. AutoGen Core routes it directly to the receiver in the two-agent baseline. In the three-agent version, the sender addresses the mediator, which forwards a verified query to the receiver. The console prints each hop and the reply.

## Run locally

From the repository root, `cd initial_test` and run `./run_demo.sh` with Python 3.12 installed. The launcher saves each run's output and errors to a timestamped file in `outputs/` and prints the log path in the terminal. A run with both routes keeps both results together in one file.

The default run looks up `CIHI-LTC-FALLS`, a local entry based on CIHI’s published 2024–2025 Canadian long-term care falls result (17%). Both routes return the same excerpt. The mediator changes “Hey, could you please tell me the 2024–2025 long-term care falls rate?” to “Tell me the 2024–2025 long-term care falls rate?” and preserves the index and 20-word answer limit. Use `--index CIHI-LTC-ANTIPSYCHOTICS` with a matching `--question` to try the other entry. The following command shows the mediator rejecting an unknown index before the receiver is called.

```bash
./run_demo.sh --route mediated --index UNKNOWN
```

## Try the model

Run the demo in live mode with a Cohere API key. If `COHERE_API_KEY` is absent, the demo prompts for the key and hides your input. Export `COHERE_API_KEY` to reuse the key across runs in the same shell.

```bash
./run_demo.sh --live
./run_demo.sh --route mediated --rewrite model
./run_demo.sh --live --rewrite model
```

The default live model is `command-r-08-2024`. Set `COHERE_MODEL` to a model available on your key if needed. `--live` turns on model answers. `--rewrite model` turns on model rewriting and can run with the local receiver. The last command makes three model calls across both routes for a valid query.

## Mediator design

1. **Explicit check.** Require a known document index, a limit of 1–120 answer words, and a nonempty question up to 500 characters.
2. **Open-ended transformation.** The default rules collapse whitespace and remove one recognizable filler phrase. The optional Cohere Command R rewriter edits only the question. The document index and answer limit are carried in separate fields and stay outside the rewrite call.
3. **Verifier.** Reject a candidate with altered protected fields, an empty question, or an overlong question. Verification covers structural invariants. A model rewrite can still alter the intended meaning, so compare the printed questions when evaluating that mode.

The CLI is the human entry point. This experiment mediates the outbound agent-to-agent query. Responses return through the same call chain without a rewrite. The index is a toy document identifier, and both receivers look it up in the same in-memory dictionary.

## References

- [AutoGen Core direct messaging](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/design-patterns/concurrent-agents.html#direct-messages)
- [Input and output guardrails](https://openai.github.io/openai-agents-js/guides/guardrails/)
- [Cohere OpenAI compatibility](https://docs.cohere.com/docs/compatibility-api)

- [CIHI Long-term care dashboard](https://www.cihi.ca/en/dashboards/long-term-care)
- [CIHI Keeping residents safe in long-term care](https://www.cihi.ca/en/aging-with-dignity-using-data-to-strengthen-long-term-care-for-canadians/keeping-residents-safe-in-long-term-care)
