import os
import tempfile
import unittest

from vaani.asr import Segment
from vaani.embed import HashingEmbedder
from vaani.pipeline import Pipeline, to_markdown
from vaani.store import NoteStore
from vaani.summarize import ExtractiveSummarizer, LocalLLMSummarizer


class FakeASR:
    def transcribe(self, path, language=None):
        segs = [
            Segment(0, 5, "Welcome everyone to the quarterly planning meeting."),
            Segment(5, 10, "We agreed to ship the beta on Friday."),
            Segment(10, 15, "Priya will write the release notes by Thursday."),
            Segment(15, 20, "The budget for cloud costs is too high this quarter."),
            Segment(20, 25, "Ravi needs to renegotiate the hosting contract."),
            Segment(25, 30, "Thanks all, meeting closed."),
        ]
        return segs, "en"


def make_pipeline(tmp):
    emb = HashingEmbedder()
    return Pipeline(FakeASR(), emb, ExtractiveSummarizer(emb), NoteStore(os.path.join(tmp, "t.db")))


class PipelineTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp)
            nid = p.process("standup.wav", "Standup")
            note = p.store.get_note(nid)
            self.assertEqual(note["language"], "en")
            self.assertTrue(note["summary"]["summary"])
            self.assertTrue(any("Priya" in a for a in note["summary"]["action_items"]))
            self.assertIn("## Action items", to_markdown(note))

    def test_semantic_search_ranks_relevant_chunk_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp)
            p.process("a.wav", "A")
            top = p.search("hosting contract cloud budget", k=1)[0]
            self.assertIn("hosting", top["text"].lower())

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp)
            nid = p.process("a.wav")
            p.store.delete_note(nid)
            self.assertIsNone(p.store.get_note(nid))
            self.assertEqual(p.search("anything"), [])

    def test_llm_falls_back_when_server_down(self):
        emb = HashingEmbedder()
        s = LocalLLMSummarizer("http://127.0.0.1:9/v1", timeout=1, fallback=ExtractiveSummarizer(emb))
        out = s.summarize("We will ship on Friday. Everyone agreed to the plan.")
        self.assertIn("extractive", out["note"])


class WebTests(unittest.TestCase):
    def test_routes(self):
        from app import create_app
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(make_pipeline(tmp))
            c = app.test_client()
            self.assertEqual(c.get("/").status_code, 200)
            r = c.post("/upload", data={"audio": (__import__("io").BytesIO(b"x"), "demo.wav")},
                       content_type="multipart/form-data")
            self.assertEqual(r.status_code, 302)
            self.assertEqual(c.get("/note/1").status_code, 200)
            self.assertEqual(c.get("/note/1/export.md").status_code, 200)
            self.assertEqual(c.get("/search?q=friday").status_code, 200)


if __name__ == "__main__":
    unittest.main()
