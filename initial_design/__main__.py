"""Run from the repository root with python -m initial_design."""
import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from getpass import getpass
from pathlib import Path

from .agents import Answer, run
from .trends import RateTable


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask three AutoGen agents about an annual percentage series.")
    parser.add_argument("question", nargs="?", default="Show the falls rate trend from 2021 to 2024.")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("synthetic_rates.csv"))
    parser.add_argument("--max-queries", type=int, default=2, help="Maximum data lookups per question.")
    parser.add_argument("--trace", action="store_true", help="Print each public blackboard post.")
    parser.add_argument("--live", action="store_true", help="Interpret free-form questions with Cohere.")
    args = parser.parse_args()
    if args.max_queries < 1:
        parser.error("--max-queries must be positive")
    try:
        table = RateTable(args.data)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    client = None
    if args.live:
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        key = os.getenv("COHERE_API_KEY", "").strip()
        if not key and sys.stdin.isatty():
            key = getpass("Paste your Cohere API key (input hidden), then press Return: ").strip()
        if not key:
            parser.error("A Cohere API key is required. Run --live in a terminal or set COHERE_API_KEY.")
        client = OpenAIChatCompletionClient(
            model=os.getenv("COHERE_MODEL", "command-r-08-2024"), api_key=key,
            base_url="https://api.cohere.ai/compatibility/v1", timeout=60, max_retries=0,
            model_info=dict(family="unknown", vision=False, function_calling=False,
                            json_output=False, structured_output=False),
        )
    try:
        board = await run(args.question, table, max_queries=args.max_queries, client=client)
    finally:
        if client is not None:
            await client.close()
    if args.trace:
        for i, post in enumerate(board.posts, 1):
            payload = json.dumps(asdict(post.content), default=str, ensure_ascii=False)
            print(f"{i:02} {post.author} {type(post.content).__name__} {payload}")
    answer = board.latest(Answer)
    print(f"[{answer.status}] {args.data.name}\n{answer.text}")


if __name__ == "__main__":
    asyncio.run(main())
