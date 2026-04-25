import unittest

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

if __name__ == "__main__":
    unittest.main()
