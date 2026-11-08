import unittest
from unittest.mock import Mock

from src.generation.answerer import Answerer
from src.generation.graduation_credits import lookup_graduation_total


QUESTION = "2024级计算机科学与技术专业毕业需要修满多少学分？"
PLAN = "2024计算机科学与技术培养计划.pdf"
CHUNKS = [
    {"source_file": PLAN, "text": "专业必修课程最低要求18学分。"},
    {"source_file": "2024软件工程培养计划.pdf", "text": "毕业学分要求：168学分。"},
    {"source_file": PLAN, "text": "毕业学分要求：165.5学分。第二课堂学分要求：6学分。"},
]


class GraduationCreditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.retriever = Mock()
        self.llm = Mock()
        self.answerer = Answerer(self.retriever, self.llm, policy_chunks=CHUNKS)

    async def test_sync_and_stream_answer_from_the_requested_plan(self):
        result = self.answerer.answer(QUESTION)
        self.assertIn("毕业总学分为 165.5 学分", result["answer"])
        self.assertEqual([c["source"].source_file for c in result["citations"]], [PLAN])
        self.assertFalse(result["fallback"].should_fallback)
        events = [event async for event in self.answerer.astream_answer(QUESTION)]
        self.assertEqual([event["type"] for event in events], ["token", "done"])
        self.assertEqual(events[-1]["answer"], result["answer"])
        self.assertEqual(events[-1]["citations"], result["citations"])
        self.retriever.invoke.assert_not_called()
        self.llm.invoke.assert_not_called()
        self.llm.astream.assert_not_called()

    async def test_missing_year_or_conflicting_total_falls_back(self):
        missing_year = QUESTION.replace("2024", "2025")
        self.assertTrue(self.answerer.answer(missing_year)["fallback"].should_fallback)
        self.assertEqual(
            [event["type"] async for event in self.answerer.astream_answer(missing_year)],
            ["fallback"],
        )
        conflict = CHUNKS + [{"source_file": PLAN, "text": "毕业总学分要求：166学分。"}]
        self.assertEqual(lookup_graduation_total(QUESTION, conflict), ("", []))
        self.assertEqual(lookup_graduation_total(QUESTION, CHUNKS[:2]), ("", []))
        self.retriever.invoke.assert_not_called()

    def test_other_credit_questions_keep_the_normal_retrieval_path(self):
        question = "计算机科学与技术专业的专业必修课程需要多少学分？"
        self.retriever.invoke.return_value = []
        self.answerer.answer(question)
        self.retriever.invoke.assert_called_once_with(question)
