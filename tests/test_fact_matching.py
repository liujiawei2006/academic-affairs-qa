import json
import unittest
from pathlib import Path

from langchain_core.documents import Document

from run_hallucination_eval import check_condition_in_answer, check_value_in_answer
from src.generation.verifier import verify


class FactMatchingTests(unittest.TestCase):
    def test_wrong_credit_number_or_unit_is_not_evidence(self):
        for answer, evidence in (
            ("毕业需要6学分。", "毕业总计166学分。"),
            ("需要6学分。", "最低要求16.6学分。"),
            ("需要20学分。", "奖学金金额20元。"),
            ("毕业不需要166学分。", "毕业需要166学分。"),
        ):
            with self.subTest(answer=answer, evidence=evidence):
                self.assertFalse(verify(answer, [Document(page_content=evidence)]).is_clean)

    def test_credit_scope_uses_field_and_source_major(self):
        software = Document(
            page_content="毕业学分要求：166 学分。第二课堂学分要求：7 学分。",
            metadata={"source_file": "2024软件工程培养计划.pdf"},
        )
        computing = Document(
            page_content="毕业学分要求：１６６ 学分。第二课堂学分要求：6 学分。",
            metadata={"source_file": "2024计算机科学与技术培养计划.pdf"},
        )
        cases = (
            ("软件工程专业毕业需修满166学分[1]。", True),
            ("软件工程第二课堂要求166学分[1]。", False),
            ("计算机科学与技术第二课堂要求7学分。", False),
            ("计算机科学与技术毕业要求166学分，第二课堂要求7学分。", False),
            ("计算机科学与技术毕业要求166学分，第二课堂要求6学分。", True),
            ("计算机科学与技术专业毕业需修满166学分[1]。", False),
            ("计算机科学与技术专业毕业需修满166学分[2]。", True),
            ("计算机科学与技术和软件工程的毕业总学分相同，均为166学分[1][2]。", True),
            ("计算机科学与技术和软件工程的毕业总学分相同，均为166学分[1]。", False),
        )
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(verify(answer, [software, computing]).is_clean, expected)

    def test_value_scoring_requires_number_unit_and_marked_scope(self):
        self.assertTrue(check_value_in_answer("6学分", "需要６ 学分。"))
        self.assertFalse(check_value_in_answer("6学分", "需要166学分。"))
        self.assertFalse(check_value_in_answer("20学分", "金额为20元。"))
        self.assertFalse(check_value_in_answer("3.8", "绩点为13.8。"))
        self.assertFalse(check_value_in_answer("20学分", "不需要20学分。"))
        self.assertFalse(check_value_in_answer("软件工程专业必修20学分", "网络工程专业必修20学分。"))
        self.assertFalse(check_value_in_answer("软件工程毕业166学分", "软件工程第二课堂166学分。"))

    def test_condition_scoring_and_verification_distinguish_negation(self):
        condition = "必须通过资格审核"
        self.assertTrue(check_condition_in_answer(condition, "申请人必须通过资格审核。"))
        self.assertFalse(check_condition_in_answer(condition, "申请人不必通过资格审核。"))
        self.assertFalse(check_condition_in_answer(condition, "并非必须通过资格审核。"))
        self.assertFalse(check_condition_in_answer("通过资格审核", "未通过资格审核。"))
        evidence = [Document(page_content="申请人不必通过资格审核。")]
        self.assertFalse(verify(condition, evidence).is_clean)
        self.assertTrue(verify("不必通过资格审核。", evidence).is_clean)

    def test_corrected_gold_answers_match_their_scoring_labels(self):
        dataset = Path(__file__).with_name("golden_dataset_answer.jsonl")
        ids = {13, 57, 96, 102, 104}
        with dataset.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
        rows = [row for row in rows if row["id"] in ids]
        self.assertEqual({row["id"] for row in rows}, ids)
        for row in rows:
            with self.subTest(id=row["id"]):
                self.assertIn("166 学分", row["answer"])
                self.assertNotIn("160 学分", row["answer"])
                self.assertTrue(all(check_value_in_answer(v, row["answer"]) for v in row["expected_values"]))
