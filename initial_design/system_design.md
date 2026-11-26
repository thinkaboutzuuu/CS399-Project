# Initial design for blackboard-mediated trend queries

A user asks how a measured rate changed over time. Three Microsoft AutoGen Core agents turn the question into a data lookup, check whether the evidence supports a trend, and return the result with its observations. The bundled CSV contains **synthetic annual long-term-care rates**. The values illustrate how a trend can reverse within a period. They are invented for this experiment.

## Run the prototype

From the repository root, using Python 3.12:

```sh
python3.12 -m venv initial_design/.venv
source initial_design/.venv/bin/activate
python -m pip install -r initial_design/requirements.txt
python -m initial_design --trace
```

If the earlier mediator demo is already installed, its environment can also run the new system directly from the repository root:

```sh
initial_test/mediator_demo/.venv/bin/python -m initial_design --trace
```

The default run asks for the falls rate trend from 2021 to 2024. Compare it with an endpoint question, or limit the number of Query Agent turns to see how the evaluation changes:

```sh
python -m initial_design "What was the net change in the falls rate between 2021 and 2024?" --trace
python -m initial_design "Show the antipsychotic use rate trend from 2021 to 2024."
python -m initial_design --max-queries 1 --trace
```

The offline parser recognizes the question forms above with any metric in the input CSV. The optional `--live` flag uses the repo’s Cohere-compatible AutoGen model client to interpret free-form questions. In a terminal, paste your Cohere API key at the hidden prompt for a single run. Set `COHERE_API_KEY` in your environment to reuse it across runs or run noninteractively. You can select a model with `COHERE_MODEL`. Live interpretation sends the question and available metric names to Cohere. Data retrieval, evaluation, and answer formatting run locally in both modes.

```sh
python -m initial_design --live --trace "Did the falls rate fall consistently from 2021 through 2024?"
```

## Reading Gwen’s diagram

| Sketch steps | Implemented exchange |
| --- | --- |
| 1–2 · User → Interaction → Blackboard | The user’s text opens a fresh board. Interaction reads it and posts a structured query containing one metric, an inclusive year range, and the requested analysis. An unsupported request produces a clarification. |
| 3–5 · Blackboard → Query ↔ Data | The control loop activates Query with the current board. Query reads the requested years from the CSV-backed table. It alone has access to the observations in that table. |
| 6–7 · Query → Blackboard → Evaluation | Query posts both its claim and the observations used to compute it. Evaluation reads the claim, query, and evidence together. |
| 8 · Evaluation → Blackboard | Evaluation posts acceptance, a request for annual evidence, or an explanation of insufficient evidence. A request for more evidence makes Query eligible to run again. |
| 9–10 · Blackboard → Interaction → User | Interaction reads the final verdict. On acceptance it reports the checked claim and the cited year–rate pairs. Otherwise it returns the reason the system could not answer. |

The central design choice is that an agent can see *why* another lookup was requested. The first claim, the evaluator’s feedback, and the replacement evidence all survive on the same board. The retry is my interpretation of the feedback arrow in [Gwen’s original sketch](../Gwen%27s_init_design.png).

## Blackboard and control unit

`agents.py` registers Interaction, Query, and Evaluation as separate `RoutedAgent` types in a `SingleThreadedAgentRuntime`. Each call to `run()` owns a fresh runtime and board. The board holds an ordered tuple of attributed posts. Each role receives the full immutable snapshot through AutoGen’s `send_message`, reads it, and returns one typed post. Only the control loop appends that post. The roles have no peer addresses or private conversation histories.

`next_role()` selects the next role from the latest post. A query activates Query, evidence activates Evaluation, and a request for annual evidence reactivates Query. A terminal verdict activates Interaction. The loop ends with an `Answer`. The default budget is two data lookups. The first reads endpoints and the second, if requested, reads the entire interval. After that the evaluator has all available rows and can accept the claim or explain a gap. The trace prints the chronological board, including revisions.

This implements the paper’s shared public working memory and its cycle of selecting a contributor from the current board state. I used Gwen’s three fixed roles and a deterministic control unit because evidence coverage gives this task a testable stopping condition. The paper also explores generated experts, private debate spaces, LLM scheduling, and multiple ways to select a final answer. Those are outside this prototype. The linked third-party implementation served as a reference for the distinction between the blackboard and scheduler. The code here was written for AutoGen. [LbMAS, §3](https://arxiv.org/html/2507.01701v1#S3) · [Community implementation](https://github.com/lightflow16/sde_project_bMAS)

## What counts as a trend

`trends.py` treats a **net change** as the end rate minus the start rate, in percentage points. A **trend** also asks about the path between them. For a trend, the evaluator requires one observation for every requested year, including both endpoints. It checks the posted difference and annual pattern against those observations. Rates use decimal arithmetic. Direction is descriptive and comes from consecutive annual changes. It carries no implication of statistical significance or causation.

In the default run, the endpoints are 18.0% and 15.0%. They support a net decrease of 3.0 percentage points. They leave open whether the rate fell every year. Evaluation requests annual evidence, and Query posts 18.0%, 16.5%, 17.0%, and 15.0%. The rise in 2023 explains the accepted description **changed direction**. For a net-change request, those interior years are unnecessary. For a two-year trend spanning consecutive years, the endpoints already give full annual coverage.

A missing endpoint ends the run as `insufficient`. A missing interior year discovered on the full read also returns `insufficient`. With a one-lookup budget the system reports when it could only check the endpoints. It never fills gaps by interpolation.

## Data boundary and verification

Supply another CSV with `--data path/to/rates.csv`. The exact header is `metric,year,rate`. Each row represents one annual percentage, from 0 to 100 inclusive, for one metric and four-digit year. The loader normalizes metric names to lowercase, rejects duplicate metric–year pairs, and validates values before the agents start. The example assumes one consistent population and measurement definition per metric. The evaluator checks consistency with the posted observations. It cannot establish source quality or whether a denominator changed.

The offline path uses a small grammar and returns `clarify` for unsupported wording. In live mode the model proposes a query and the same validator checks its metric, year order, and analysis kind. A valid query can still misinterpret the user’s wording. The trace exposes the selected query. Board state and the data index are in memory for the duration of one run.

Run the regression suite from the repository root:

```sh
python -m unittest -v initial_design.test_system
```

The tests exercise the AutoGen runtime, evidence-request loop, endpoint-only and consecutive-year fast paths, missing data, exhausted budget, unsupported filters, model response validation, and rejection of an inconsistent claim. The model-path test uses AutoGen’s replay client so it runs without credentials.
