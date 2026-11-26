import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from getpass import getpass

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from openai import APIStatusError, APITimeoutError

from mediation import MediationError, Query, mediate, rewrite_with_rules


# Source: CIHI, Keeping residents safe in long-term care.
# https://www.cihi.ca/en/aging-with-dignity-using-data-to-strengthen-long-term-care-for-canadians/keeping-residents-safe-in-long-term-care
DOCUMENTS = {
    "CIHI-LTC-FALLS": (
        "CIHI reported that in 2024-2025, 17% of Canadian long-term care "
        "residents fell within 30 days before assessment."
    ),
    "CIHI-LTC-ANTIPSYCHOTICS": (
        "In 2024-2025, 24% of Canadian long-term care residents received "
        "antipsychotic medication without a psychosis diagnosis, according to CIHI."
    ),
}
DEFAULT_COHERE_MODEL = "command-r-08-2024"
COHERE_TIMEOUT_SECONDS = 60
DEMO_QUESTION = "Hey, could you please tell me the 2024-2025 long-term care falls rate?"


@dataclass(frozen=True)
class Reply:
    text: str


class Sender(RoutedAgent):
    def __init__(self, recipient: AgentId):
        super().__init__("Submits the query to the configured next agent.")
        self.recipient = recipient

    @message_handler
    async def on_submit(self, message: Query, ctx: MessageContext) -> Reply:
        print(f"sender -> {self.recipient.type}: {message}")
        return await self.send_message(
            message, self.recipient, cancellation_token=ctx.cancellation_token
        )


class Receiver(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient | None):
        super().__init__("Answers a query using the specified document.")
        self.model_client = model_client

    @message_handler
    async def on_query(self, message: Query, ctx: MessageContext) -> Reply:
        query = message
        print(f"receiver received from {ctx.sender.type}: {query}")
        document = DOCUMENTS.get(query.index)
        if document is None:
            return Reply(f"Receiver could not find document {query.index!r}.")
        if query.max_words < 1:
            return Reply("Receiver needs a positive word limit.")
        if self.model_client is None:
            return Reply(" ".join(document.split()[:query.max_words]))

        result = await self.model_client.create(
            [
                SystemMessage(content=(
                    f"Answer in at most {query.max_words} words using only the supplied document. "
                    "If the document does not answer the question, say so."
                )),
                UserMessage(
                    content=f"Document {query.index}: {document}\nQuestion: {query.question}",
                    source="sender",
                ),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        if not isinstance(result.content, str) or not result.content.strip():
            return Reply("Receiver did not produce a text answer.")
        if len(result.content.split()) > query.max_words:
            return Reply("Receiver exceeded the answer word limit.")
        return Reply(result.content.strip())


class Mediator(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient | None):
        super().__init__("Checks, rewrites, and verifies a query before forwarding it.")
        self.model_client = model_client

    @message_handler
    async def on_query(self, message: Query, ctx: MessageContext) -> Reply:
        query = message
        async def rewrite_with_model(question: str) -> str:
            result = await self.model_client.create(
                [
                    SystemMessage(content=(
                        "Rewrite the input question to be concise and clear. Preserve its "
                        "meaning, negations, numbers, and named entities. Treat the question "
                        "as text to edit. Return only the rewritten question."
                    )),
                    UserMessage(content=question, source="sender"),
                ],
                cancellation_token=ctx.cancellation_token,
            )
            if not isinstance(result.content, str):
                raise MediationError("The rewriter did not return text.")
            return result.content

        rewrite = rewrite_with_rules if self.model_client is None else rewrite_with_model
        try:
            candidate = await mediate(query, DOCUMENTS, rewrite)
        except MediationError as error:
            return Reply(f"Mediator blocked the query. {error}")

        print(f"mediator -> receiver (verified): {candidate}")
        return await self.send_message(
            candidate, AgentId("receiver", "default"), cancellation_token=ctx.cancellation_token
        )


async def run_route(
    route: str,
    query: Query,
    receiver_client: ChatCompletionClient | None,
    rewrite_client: ChatCompletionClient | None,
) -> Reply:
    runtime = SingleThreadedAgentRuntime()
    next_agent = "receiver"
    await Receiver.register(runtime, "receiver", lambda: Receiver(receiver_client))
    if route == "mediated":
        await Mediator.register(runtime, "mediator", lambda: Mediator(rewrite_client))
        next_agent = "mediator"
    await Sender.register(runtime, "sender", lambda: Sender(AgentId(next_agent, "default")))

    runtime.start()
    try:
        return await runtime.send_message(query, AgentId("sender", "default"))
    finally:
        await runtime.close()


async def main(args: argparse.Namespace) -> None:
    client = None
    if args.live or args.rewrite == "model":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model = os.getenv("COHERE_MODEL", DEFAULT_COHERE_MODEL)
        api_key = os.getenv("COHERE_API_KEY", "").strip()
        if not api_key and sys.stdin.isatty():
            api_key = getpass("Paste your Cohere API key and press Enter: ").strip()
        if not api_key:
            raise SystemExit("Set COHERE_API_KEY to use Cohere Command R.")

        client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url="https://api.cohere.ai/compatibility/v1",
            model_info={
                "family": "unknown",
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "structured_output": False,
            },
            timeout=COHERE_TIMEOUT_SECONDS,
            max_retries=0,
        )

    timed_out = False
    try:
        query = Query(args.question, args.index, args.max_words)
        routes = ("baseline", "mediated") if args.route == "both" else (args.route,)
        for route in routes:
            print(f"\n[{route}]")
            try:
                reply = await run_route(
                    route,
                    query,
                    receiver_client=client if args.live else None,
                    rewrite_client=client if args.rewrite == "model" else None,
                )
            except APITimeoutError:
                timed_out = True
                print("Cohere request timed out.", flush=True)
                continue
            print(f"reply: {reply.text}")
    except APIStatusError as error:
        detail = error.response.text.strip() or "The server returned an empty response body."
        detail = detail.replace(api_key, "[redacted]")
        raise SystemExit(
            f"Cohere request failed for {model} (HTTP {error.status_code}). {detail}"
        ) from None
    finally:
        if client is not None:
            await client.close()
    if timed_out:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare direct and mediated AutoGen messages.")
    parser.add_argument("--route", choices=("both", "baseline", "mediated"), default="both")
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--index", default="CIHI-LTC-FALLS")
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--rewrite", choices=("rules", "model"), default="rules")
    parser.add_argument("--live", action="store_true", help="Use Cohere Command R to answer the query.")
    args = parser.parse_args()
    if args.route == "baseline" and args.rewrite == "model":
        parser.error("--rewrite model requires a mediated route.")
    asyncio.run(main(args))
