"""Offline checks: python -m unittest discover -s tests -v"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import rag

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def sample_transcript(video_id="cidPApHyXhI"):
    return rag.TranscriptData(video_id, "English", "en", False, (
        rag.Caption("Solar panels turn sunlight into electricity. " * 15, 0, 25),
        rag.Caption("Batteries store energy for the night. " * 15, 25, 25),
        rag.Caption("Wind turbines generate power using wind. " * 15, 50, 25),
    ))


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_documents = 0

    def embed_documents(self, texts):
        self.embedded_documents += len(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        text = text.lower()
        return [float(text.count(term) + 1) for term in ("solar", "batter", "wind")]


class VideoParsingTests(unittest.TestCase):
    def test_supported_youtube_links(self):
        for value in (
            "https://www.youtube.com/watch?v=cidPApHyXhI&list=test",
            "https://youtu.be/cidPApHyXhI?t=10",
            "https://m.youtube.com/watch?v=cidPApHyXhI",
            "https://www.youtube.com/shorts/cidPApHyXhI",
            "https://youtube.com/live/cidPApHyXhI",
            "https://www.youtube-nocookie.com/embed/cidPApHyXhI",
            "youtube.com/watch?v=cidPApHyXhI", "cidPApHyXhI",
        ):
            with self.subTest(value=value):
                self.assertEqual(rag.parse_video_id(value), "cidPApHyXhI")

    def test_invalid_urls_and_spoofed_hosts(self):
        for value in ("", "https://example.com/watch?v=cidPApHyXhI",
                      "https://youtube.com.example.com/watch?v=cidPApHyXhI",
                      "https://youtube.com@evil.com/watch?v=cidPApHyXhI",
                      "https://youtube.com/watch?v=short", "https://youtube.com/playlist?list=test",
                      "ftp://youtube.com/watch?v=cidPApHyXhI"):
            with self.subTest(value=value), self.assertRaises(rag.AppError):
                rag.parse_video_id(value)

    def test_timestamps(self):
        self.assertEqual(rag.timestamp(65.9), "1:05")
        self.assertEqual(rag.timestamp(3605), "1:00:05")


class TranscriptTests(unittest.TestCase):
    def test_chunks_preserve_text_and_timestamps(self):
        data = sample_transcript()
        documents = rag.transcript_documents(data)
        self.assertGreater(len(documents), 1)
        for doc in documents:
            start = doc.metadata["start_index"]
            self.assertEqual(data.text[start:start + len(doc.page_content)], doc.page_content)
            self.assertGreaterEqual(doc.metadata["start"], 0)
            self.assertGreater(doc.metadata["end"], doc.metadata["start"])
            self.assertEqual(doc.metadata["video_id"], data.video_id)
            self.assertIn("&t=", doc.metadata["source"])

    def test_collection_changes_when_video_model_or_transcript_changes(self):
        first = sample_transcript()
        collection = rag.collection_name(first, "model-a")
        self.assertNotEqual(collection, rag.collection_name(sample_transcript("abcdefghijk"), "model-a"))
        self.assertNotEqual(collection, rag.collection_name(first, "model-b"))
        changed = rag.TranscriptData(first.video_id, "English", "en", False, (rag.Caption("New content", 0, 4),))
        self.assertNotEqual(collection, rag.collection_name(changed, "model-a"))

    def test_language_fallback(self):
        track = SimpleNamespace(is_generated=False, language_code="hi")
        track.fetch = MagicMock(return_value=SimpleNamespace(
            language="Hindi", language_code="hi", is_generated=False,
            __iter__=lambda self: iter([]),
        ))
        fetched = MagicMock()
        fetched.language, fetched.language_code, fetched.is_generated = "Hindi", "hi", False
        fetched.__iter__.return_value = iter([SimpleNamespace(text="hello &amp; goodbye", start=0, duration=5)])
        track.fetch.return_value = fetched
        available = MagicMock()
        available.find_transcript.side_effect = rag.NoTranscriptFound("cidPApHyXhI", ["en"], "")
        available.__iter__.return_value = iter([track])
        with patch("rag.YouTubeTranscriptApi") as api:
            api.return_value.list.return_value = available
            result = rag.fetch_transcript("cidPApHyXhI")
        self.assertEqual(result.language_code, "hi")
        self.assertEqual(result.text, "hello & goodbye")

    def test_disabled_transcript_has_actionable_error(self):
        with patch("rag.YouTubeTranscriptApi") as api:
            api.return_value.list.side_effect = rag.TranscriptsDisabled("cidPApHyXhI")
            with self.assertRaisesRegex(rag.AppError, "disabled"):
                rag.fetch_transcript("cidPApHyXhI")


class RetrievalTests(unittest.TestCase):
    def test_real_chroma_persistence_reuse_and_video_isolation(self):
        embeddings = LocalEmbeddings()
        settings = rag.Settings("fake-openrouter", "fake-google")
        with tempfile.TemporaryDirectory() as directory, patch("rag.OpenAIEmbeddings", return_value=embeddings), patch("rag.ChatGoogleGenerativeAI", return_value=RunnableLambda(lambda _: "Grounded answer [1].")):
            first = rag.build_index(sample_transcript(), settings, Path(directory))
            count = embeddings.embedded_documents
            second = rag.build_index(sample_transcript(), settings, Path(directory))
            self.assertEqual(embeddings.embedded_documents, count)
            other = rag.build_index(sample_transcript("abcdefghijk"), settings, Path(directory))
            self.assertEqual(len(first.vector_store.get(include=[])['ids']), len(first.documents))
            result = second.answer("How do batteries work?")
            self.assertTrue(result.sources)
            self.assertTrue(all(doc.metadata['video_id'] == "cidPApHyXhI" for doc in result.sources))
            self.assertNotEqual(first.vector_store._collection.name, other.vector_store._collection.name)
            # Release all persistent clients before Windows removes the temp database.
            first.vector_store._client.close()
            second.vector_store._client.close()
            other.vector_store._client.close()

    def test_followup_is_rewritten_before_retrieval(self):
        store = MagicMock()
        docs = rag.transcript_documents(sample_transcript())
        store.similarity_search.return_value = docs[:1]
        calls = []
        def respond(prompt):
            calls.append(prompt.to_messages())
            return "How do batteries store energy?" if len(calls) == 1 else "They store energy [1]."
        engine = rag.VideoRAG(sample_transcript(), docs, store, RunnableLambda(respond), "test")
        engine.answer("How do they do that?", [{"role": "user", "content": "Tell me about batteries."}])
        self.assertEqual(store.similarity_search.call_args.args[0], "How do batteries store energy?")
        self.assertIn("not evidence", calls[-1][0].content)

    def test_no_results_returns_unknown_without_model_call(self):
        store = MagicMock()
        store.similarity_search.return_value = []
        llm = RunnableLambda(lambda _: self.fail("Model shouldn't run without context"))
        engine = rag.VideoRAG(sample_transcript(), rag.transcript_documents(sample_transcript()), store, llm, "test")
        self.assertEqual(engine.answer("Who won a football game?").text, rag.UNKNOWN)

    def test_summary_covers_all_chunks_including_the_last(self):
        docs = [Document(page_content=f"UNIQUE-PASSAGE-{i} " + "x" * 900,
                         metadata={"start": i * 10, "end": i * 10 + 10, "source": "https://youtube.com"})
                for i in range(45)]
        contexts = []
        def respond(prompt):
            contexts.append(prompt.to_messages()[-1].content)
            return "Summary notes [1]."
        store = MagicMock()
        engine = rag.VideoRAG(sample_transcript(), docs, store, RunnableLambda(respond), "test")
        result = engine.answer("Summarize the video")
        self.assertEqual(result.sources, docs)
        store.similarity_search.assert_not_called()
        mapped_context = "\n".join(contexts[:-1])
        for i in range(45):
            self.assertIn(f"UNIQUE-PASSAGE-{i} ", mapped_context)
        self.assertIn("[45]", mapped_context)


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENROUTER_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def app(self):
        return AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    def test_empty_page_and_missing_keys(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Your video. Your questions.")
        self.assertTrue(app.chat_input[0].disabled)
        next(widget for widget in app.text_input if widget.label == "YouTube URL").input("https://youtu.be/cidPApHyXhI")
        next(button for button in app.button if button.label == "Load video").click().run()
        self.assertFalse(app.exception)
        self.assertIn("both", app.error[0].value)

    def test_server_keys_are_never_populated_into_browser_inputs(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "server-openrouter-secret", "GOOGLE_API_KEY": "server-google-secret"}):
            app = self.app()
            for label in ("OpenRouter API key", "Google API key"):
                self.assertEqual(next(widget for widget in app.text_input if widget.label == label).value, "")
            self.assertNotIn("server-openrouter-secret", str(app))
            self.assertNotIn("server-google-secret", str(app))

    def test_public_mode_requires_visitor_keys_even_with_server_keys(self):
        with patch.dict(os.environ, {"REQUIRE_USER_KEYS": "true", "OPENROUTER_API_KEY": "server-key", "GOOGLE_API_KEY": "server-key"}):
            app = self.app()
            next(widget for widget in app.text_input if widget.label == "YouTube URL").input("https://youtu.be/cidPApHyXhI")
            next(button for button in app.button if button.label == "Load video").click().run()
            self.assertFalse(app.exception)
            self.assertIn("both", app.error[0].value)

    def test_load_chat_clear_and_failed_switch_preserves_active_video(self):
        settings = rag.Settings("test-openrouter", "test-google")
        data = sample_transcript()
        engine = MagicMock()
        engine.settings_signature = settings.signature
        engine.transcript = data
        engine.documents = rag.transcript_documents(data)
        engine.answer.return_value = rag.Answer("Batteries store energy [1].", engine.documents[:1])
        with patch("rag.fetch_transcript", return_value=data) as fetch, patch("rag.build_index", return_value=engine):
            app = self.app()
            for label, value in (("OpenRouter API key", "test-openrouter"), ("Google API key", "test-google"), ("YouTube URL", "https://youtu.be/cidPApHyXhI")):
                next(widget for widget in app.text_input if widget.label == label).input(value)
            next(button for button in app.button if button.label == "Load video").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.chat_input[0].disabled)
            app.chat_input[0].set_value("What do batteries do?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 2)
            self.assertIn("Batteries", app.session_state["messages"][1]["content"])
            self.assertIn("&t=", app.session_state["messages"][1]["sources"][0]["url"])
            next(button for button in app.button if button.label == "Clear chat").click().run()
            self.assertEqual(app.session_state["messages"], [])
            next(button for button in app.button if button.label == "Summarize video").click().run()
            self.assertTrue(engine.answer.call_args.kwargs["summarize"])
            fetch.side_effect = rag.AppError("Captions are disabled for this video.")
            next(widget for widget in app.text_input if widget.label == "YouTube URL").input("https://youtu.be/abcdefghijk")
            next(button for button in app.button if button.label == "Load video").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["engine"].transcript.video_id, data.video_id)
            next(widget for widget in app.text_input if widget.label == "Gemini model").input("a-different-model").run()
            self.assertTrue(app.chat_input[0].disabled)


if __name__ == "__main__":
    unittest.main()
