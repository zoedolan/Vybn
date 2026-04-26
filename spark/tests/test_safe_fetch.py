import unittest
from pathlib import Path

from spark.harness.safe_fetch import extract_text, validate_url

class SafeFetchTests(unittest.TestCase):
    def test_rejects_http(self):
        with self.assertRaises(ValueError):
            validate_url("http://example.com")

    def test_rejects_credentials(self):
        with self.assertRaises(ValueError):
            validate_url("https://user:pass@example.com")

    def test_rejects_localhost_ip(self):
        with self.assertRaises(ValueError):
            validate_url("https://127.0.0.1")

    def test_extracts_html_text_without_scripts(self):
        html = "<html><head><title>T</title><script>evil()</script></head><body><h1>Head</h1><p>Body text</p></body></html>"
        text = extract_text(html, "text/html")
        self.assertIn("Head", text)
        self.assertIn("Body text", text)
        self.assertNotIn("evil", text)

    def test_cli_source_mentions_untrusted_output_mode(self):
        source = Path("spark/harness/safe_fetch.py").read_text()
        self.assertIn("UNTRUSTED_TEXT_WRITTEN", source)
        self.assertIn("Path(ns.out).expanduser()", source)


if __name__ == "__main__":
    unittest.main()


class SemanticWebContentTypeTests(unittest.TestCase):
    def test_json_ld_is_allowed_content_prefix(self):
        from spark.harness.safe_fetch import ALLOWED_CONTENT_PREFIXES
        self.assertTrue(any("application/ld+json".startswith(p) for p in ALLOWED_CONTENT_PREFIXES))
