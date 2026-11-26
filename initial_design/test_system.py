import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autogen_ext.models.replay import ReplayChatCompletionClient

from .agents import Answer, Evidence, Feedback, run
from .trends import Query, RateTable, evaluate


class BlackboardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.table = RateTable(Path(__file__).with_name("synthetic_rates.csv"))
        self.question = "Show the falls rate trend from 2021 to 2024."

    async def test_feedback_reopens_query_and_exposes_reversal(self):
        board = await run(self.question, self.table)
        self.assertEqual([p.author for p in board.posts],
                         ["user", "interaction", "query", "evaluation", "query", "evaluation", "interaction"])
        self.assertEqual(board.posts[3].content.verdict, "request_annual")
        self.assertEqual(board.latest(Feedback).verdict, "accept")
        self.assertIn("changed direction", board.latest(Answer).text)
        self.assertIn("-3.0 percentage points", board.latest(Answer).text)
        self.assertEqual([p.year for p in board.latest(Evidence).points], [2021, 2022, 2023, 2024])

    async def test_endpoint_question_needs_one_lookup(self):
        board = await run(self.question.replace("trend", "net change"), self.table)
        self.assertEqual(board.query_count, 1)
        self.assertEqual(board.latest(Answer).status, "accepted")
        self.assertIsNone(board.latest(Evidence).annual_pattern)

    async def test_adjacent_years_are_already_complete(self):
        board = await run(self.question.replace("2021", "2023"), self.table)
        self.assertEqual(board.query_count, 1)
        self.assertIn("fell every year", board.latest(Answer).text)

    async def test_budget_and_missing_endpoint(self):
        for question in [self.question, self.question.replace("2021", "2020")]:
            board = await run(question, self.table, max_queries=1)
            self.assertEqual(board.latest(Answer).status, "insufficient")
            self.assertEqual(board.query_count, 1)

    async def test_missing_interior_year(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gaps.csv"
            path.write_text("metric,year,rate\nfalls rate,2021,42\nfalls rate,2024,41\n")
            board = await run(self.question, RateTable(path))
        self.assertEqual(board.query_count, 2)
        self.assertIn("missing years", board.latest(Answer).text)

    async def test_unsupported_filter_requests_clarification(self):
        board = await run(self.question + " In Toronto only.", self.table)
        self.assertEqual(board.query_count, 0)
        self.assertEqual(board.latest(Answer).status, "clarify")

    async def test_model_parsing_and_validation(self):
        client = ReplayChatCompletionClient([
            '{"metric":"falls rate","start":2021,"end":2024,"kind":"trend"}',
            '{"metric":"unavailable","start":2021,"end":2024,"kind":"trend"}',
            'invalid json',
        ])
        try:
            for expected in ["accepted", "clarify", "clarify"]:
                board = await run("How did falls change over 2021–2024?", self.table, client=client)
                self.assertEqual(board.latest(Answer).status, expected)
        finally:
            await client.close()

    def test_evaluation_rejects_unsupported_claims(self):
        query = Query("falls rate", 2021, 2024, "trend")
        evidence = self.table.lookup(query, annual=True)
        for altered in [replace(evidence, change=0), replace(evidence, annual_pattern="fell every year")]:
            self.assertEqual(evaluate(query, altered, can_retry=False).verdict, "insufficient")


if __name__ == "__main__":
    unittest.main()
