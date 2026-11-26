"""AutoGen roles and the control loop for one public blackboard."""
import json
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from .trends import Evidence, Feedback, Query, RateTable, evaluate, make_query, parse_question, render


@dataclass(frozen=True)
class Question:
    text: str


@dataclass(frozen=True)
class Answer:
    status: str
    text: str


class Post(BaseModel, frozen=True):
    author: str
    content: Question | Query | Evidence | Feedback | Answer


T = TypeVar("T")


class Board(BaseModel, frozen=True):
    posts: tuple[Post, ...]
    max_queries: int

    def latest(self, kind: type[T]) -> T:
        for post in reversed(self.posts):
            if isinstance(post.content, kind):
                return post.content
        raise LookupError(f"The board has no {kind.__name__} entry.")

    @property
    def query_count(self) -> int:
        return sum(isinstance(post.content, Evidence) for post in self.posts)


class Interaction(RoutedAgent):
    def __init__(self, metrics: tuple[str, ...], client: ChatCompletionClient | None):
        super().__init__("Interprets a user's request and presents evaluated results.")
        self.metrics = metrics
        self.client = client

    async def interpret(self, question: str, ctx: MessageContext) -> Query:
        if self.client is None:
            return parse_question(question, self.metrics)
        instructions = (
            "Translate a question about one aggregate annual percentage into JSON with exactly "
            "metric, start, end, kind. Years must be explicit integers. kind must be 'trend' "
            "or 'net change'. Trends describe the annual path. Available metric names: "
            f"{json.dumps(self.metrics)}. Return null for ambiguous or unsupported questions, "
            "including unrepresented filters, forecasts, or causal explanations. Never invent "
            "a metric or year. Return just JSON."
        )
        response = await self.client.create(
            [SystemMessage(content=instructions), UserMessage(content=question, source="user")],
            cancellation_token=ctx.cancellation_token,
        )
        try:
            data = json.loads(response.content) if isinstance(response.content, str) else None
            if not isinstance(data, dict) or set(data) != {"metric", "start", "end", "kind"}:
                raise ValueError
        except ValueError:
            raise ValueError(f"Please specify one metric ({', '.join(self.metrics)}), two years, "
                             "and either a trend or a net change.") from None
        return make_query(**data, metrics=self.metrics)

    @message_handler
    async def act(self, message: Board, ctx: MessageContext) -> Post:
        board = message
        latest = board.posts[-1].content
        if isinstance(latest, Question):
            try:
                query = await self.interpret(latest.text, ctx)
            except ValueError as error:
                return Post(author=self.id.type, content=Answer("clarify", str(error)))
            return Post(author=self.id.type, content=query)

        feedback = board.latest(Feedback)
        if feedback.verdict == "accept":
            answer = Answer("accepted", render(board.latest(Query), board.latest(Evidence)))
        else:
            answer = Answer("insufficient", feedback.reason)
        return Post(author=self.id.type, content=answer)


class QueryAgent(RoutedAgent):
    def __init__(self, table: RateTable):
        super().__init__("Reads annual rates and posts the observations supporting a trend.")
        self.table = table

    @message_handler
    async def act(self, message: Board, ctx: MessageContext) -> Post:
        board = message
        annual = isinstance(board.posts[-1].content, Feedback)
        evidence = self.table.lookup(board.latest(Query), annual)
        return Post(author=self.id.type, content=evidence)


class Evaluation(RoutedAgent):
    def __init__(self):
        super().__init__("Checks the cited evidence and requests additional observations when needed.")

    @message_handler
    async def act(self, message: Board, ctx: MessageContext) -> Post:
        board = message
        feedback = evaluate(board.latest(Query), board.latest(Evidence),
                            can_retry=board.query_count < board.max_queries)
        return Post(author=self.id.type, content=feedback)


def next_role(board: Board) -> str:
    last = board.posts[-1].content
    if isinstance(last, Query) or (isinstance(last, Feedback) and last.verdict == "request_annual"):
        return "query"
    if isinstance(last, Evidence):
        return "evaluation"
    return "interaction"


async def run(question: str, table: RateTable, *, max_queries: int = 2,
              client: ChatCompletionClient | None = None) -> Board:
    if max_queries < 1:
        raise ValueError("max_queries must be positive")
    runtime = SingleThreadedAgentRuntime()
    await Interaction.register(runtime, "interaction", lambda: Interaction(tuple(table.series), client))
    await QueryAgent.register(runtime, "query", lambda: QueryAgent(table))
    await Evaluation.register(runtime, "evaluation", Evaluation)
    board = Board(posts=(Post(author="user", content=Question(question)),), max_queries=max_queries)
    runtime.start()
    try:
        while not isinstance(board.posts[-1].content, Answer):
            role = next_role(board)
            post = await runtime.send_message(board, AgentId(role, "session"))
            board = Board(posts=(*board.posts, post), max_queries=max_queries)
        return board
    finally:
        await runtime.close()
