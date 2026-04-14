"""
Basic tests for the MBE RAG Agent.
Tests are designed to run WITHOUT a live Ollama or PostgreSQL instance
by mocking those dependencies.
"""

import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Mocking heavy dependencies so tests run without GPU/network
# ─────────────────────────────────────────────────────────────────────────────
def _mock_torch():
    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return torch_mock


for mod in ["torch", "transformers", "faiss", "langchain_ollama"]:
    sys.modules.setdefault(mod, MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestLanguageDetection(unittest.TestCase):
    def setUp(self):
        from agent.helpers import detect_language
        self.detect = detect_language

    def test_spanish_detected(self):
        cases = [
            "¿Cuál es la diferencia entre sensibilidad y especificidad?",
            "qué es un ensayo clínico aleatorizado",
            "cómo se calcula el número necesario a tratar",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.detect(text), "es")

    def test_english_detected(self):
        cases = [
            "What is the sensitivity of this diagnostic test?",
            "How do I calculate number needed to treat?",
            "Explain the CONSORT checklist for RCTs.",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.detect(text), "en")


class TestExtractJson(unittest.TestCase):
    def setUp(self):
        from agent.helpers import extract_json
        self.extract = extract_json

    def test_plain_json(self):
        text = '{"intent": "MBE", "confidence": 0.95, "reasoning": "clinical trial"}'
        result = self.extract(text)
        self.assertEqual(result["intent"], "MBE")
        self.assertAlmostEqual(result["confidence"], 0.95)

    def test_json_with_markdown_fence(self):
        text = '```json\n{"intent": "NON_MBE", "confidence": 0.8}\n```'
        result = self.extract(text)
        self.assertEqual(result["intent"], "NON_MBE")

    def test_json_with_preamble(self):
        text = 'Here is the classification:\n{"intent": "MBE", "confidence": 0.7, "reasoning": "..."}'
        result = self.extract(text)
        self.assertEqual(result["intent"], "MBE")

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            self.extract("This is not JSON at all.")


class TestRejectionMessages(unittest.TestCase):
    def setUp(self):
        from agent.helpers import build_rejection_message
        self.build = build_rejection_message

    def test_spanish_rejection(self):
        msg = self.build("es", "¿Cuál es la capital de Francia?")
        self.assertIn("MBE", msg)
        self.assertIn("capital de Francia", msg)

    def test_english_rejection(self):
        msg = self.build("en", "Who won the World Cup?")
        self.assertIn("EBM", msg)
        self.assertIn("World Cup", msg)


class TestInsufficiencyMessages(unittest.TestCase):
    def setUp(self):
        from agent.helpers import build_insufficient_retrieval_message
        self.build = build_insufficient_retrieval_message

    def test_spanish_message(self):
        msg = self.build("es")
        self.assertIn("evidencia", msg)

    def test_english_message(self):
        msg = self.build("en")
        self.assertIn("evidence", msg)


class TestFormatHistory(unittest.TestCase):
    def setUp(self):
        from agent.helpers import format_history
        self.fmt = format_history

    def test_empty_history(self):
        result = self.fmt([])
        self.assertIn("No previous", result)

    def test_formats_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self.fmt(messages)
        self.assertIn("USER:", result)
        self.assertIn("ASSISTANT:", result)
        self.assertIn("Hello", result)


class TestRAGOutput(unittest.TestCase):
    def test_rag_output_schema(self):
        from agent.tools import RAGOutput, RetrievedDocument
        doc = RetrievedDocument(content="test", source="paper.pdf", page=1, score=0.85)
        output = RAGOutput(
            documents=[doc],
            scores=[0.85],
            max_score=0.85,
            sufficient=True,
        )
        d = output.model_dump()
        self.assertTrue(d["sufficient"])
        self.assertEqual(d["max_score"], 0.85)
        self.assertEqual(len(d["documents"]), 1)

    def test_insufficient_when_below_threshold(self):
        from agent.tools import RAGOutput
        output = RAGOutput(
            documents=[],
            scores=[],
            max_score=0.10,
            sufficient=False,
        )
        self.assertFalse(output.sufficient)


class TestConfigLoading(unittest.TestCase):
    def test_settings_loaded(self):
        from utils.config import settings
        self.assertIsNotNone(settings.OLLAMA_MODEL)
        self.assertGreater(settings.FAISS_TOP_K, 0)
        self.assertGreater(settings.FAISS_SIMILARITY_THRESHOLD, 0)

    def test_postgres_dsn(self):
        from utils.config import settings
        dsn = settings.postgres_dsn
        self.assertIn("postgresql", dsn)
        self.assertIn(settings.POSTGRES_USER, dsn)


class TestPromptBuilders(unittest.TestCase):
    def test_intent_prompt_contains_query(self):
        from agent.helpers import build_intent_prompt
        prompt = build_intent_prompt("RCT blinding", "USER: hello")
        self.assertIn("RCT blinding", prompt)
        self.assertIn("MBE", prompt)

    def test_rewriting_prompt_contains_pico(self):
        from agent.helpers import build_rewriting_prompt
        prompt = build_rewriting_prompt("study on aspirin", "", "es")
        self.assertIn("PICO", prompt)
        self.assertIn("English", prompt)

    def test_response_prompt_includes_context(self):
        from agent.helpers import build_response_prompt
        prompt = build_response_prompt(
            query="What is NNT?",
            context="NNT = 1 / ARR",
            history_summary="",
            language="en",
        )
        self.assertIn("NNT = 1 / ARR", prompt)
        self.assertIn("Respond in English", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
