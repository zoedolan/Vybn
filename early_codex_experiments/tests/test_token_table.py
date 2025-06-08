import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.ledger_utils import ledger_to_markdown, parse_ledger


class TestTokenTable(unittest.TestCase):
    def test_ledger_to_markdown_contains_header(self):
        tokens = parse_ledger('token_and_jpeg_info')
        table = ledger_to_markdown(tokens)
        self.assertIn('| Token | Supply | Price | Lock | Address |', table)
        self.assertIn('SERAPH', table)


if __name__ == '__main__':
    unittest.main()
