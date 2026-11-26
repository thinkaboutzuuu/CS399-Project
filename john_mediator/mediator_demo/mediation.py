from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, replace


MAX_QUESTION_LENGTH = 500
MAX_ANSWER_WORDS = 120


@dataclass(frozen=True)
class Query:
    question: str
    index: str
    max_words: int


class MediationError(ValueError):
    pass


def explicit_check(query: Query, known_indexes: Collection[str]) -> None:
    if query.index not in known_indexes:
        raise MediationError(f"Unknown document index {query.index!r}.")
    if not 1 <= query.max_words <= MAX_ANSWER_WORDS:
        raise MediationError(f"The answer limit must be between 1 and {MAX_ANSWER_WORDS} words.")
    if not query.question.strip() or len(query.question) > MAX_QUESTION_LENGTH:
        raise MediationError("The question must be nonempty and at most 500 characters.")


async def rewrite_with_rules(question: str) -> str:
    question = " ".join(question.split())
    prefix = "hey, could you please "
    if question.casefold().startswith(prefix):
        question = question[len(prefix):]
        question = question[:1].upper() + question[1:]
    return question


def verify(original: Query, candidate: Query) -> None:
    if (candidate.index, candidate.max_words) != (original.index, original.max_words):
        raise MediationError("The rewrite changed the document index or answer limit.")
    if not candidate.question.strip() or len(candidate.question) > MAX_QUESTION_LENGTH:
        raise MediationError("The rewrite must be nonempty and at most 500 characters.")


async def mediate(
    query: Query,
    known_indexes: Collection[str],
    rewrite: Callable[[str], Awaitable[str]],
) -> Query:
    explicit_check(query, known_indexes)
    question = await rewrite(query.question)
    candidate = replace(query, question=question.strip())
    verify(query, candidate)
    return candidate
