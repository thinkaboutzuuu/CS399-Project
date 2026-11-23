# CS399 Mediator Prototype

This Python demo compares direct agent-to-agent messaging with a route that checks and rewrites the query before forwarding it to the receiver.

## Run locally

Install Python 3.12, then run this command from the repository root.

```bash
./initial_test/run_demo.sh
```

The launcher creates a virtual environment, installs dependencies on the first run, and writes a log under `initial_test/outputs/`. The default demo runs without an API key. See the [demo guide](initial_test/mediator_demo/README.md) for route options and optional Cohere integration.
