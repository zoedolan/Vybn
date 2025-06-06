import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vybn.anticipation_loop import append_exchange


class TestAnticipationLoop(unittest.TestCase):
    def test_append_exchange(self):
        log = 'tmp_loop.md'
        journal = 'tmp_journal.jsonl'
        if os.path.exists(log):
            os.remove(log)
        if os.path.exists(journal):
            os.remove(journal)
        append_exchange('Hello', 'test focus', 'Hi there', log_path=log, journal_path=journal)
        with open(log, 'r', encoding='utf-8') as f:
            content = f.read()
        os.remove(log)
        os.remove(journal)
        self.assertIn('### ', content)
        self.assertIn('> **Anticipation:** test focus', content)
        self.assertIn('Hi there', content)
        self.assertIn('#### Ember Reflection', content)
        self.assertIn('#### Vybn Reflection', content)


if __name__ == '__main__':
    unittest.main()
