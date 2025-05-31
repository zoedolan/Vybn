import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vybn.utils import write_colored


class TestWriteColored(unittest.TestCase):
    def test_default_green(self):
        buf = io.StringIO()
        sys.stdout = buf
        try:
            write_colored('hello')
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(buf.getvalue(), '\033[32mhello\033[0m\n')

    def test_error_red(self):
        buf = io.StringIO()
        sys.stdout = buf
        try:
            write_colored('oops', is_error=True)
        finally:
            sys.stdout = sys.__stdout__
        self.assertEqual(buf.getvalue(), '\033[31moops\033[0m\n')


if __name__ == '__main__':
    unittest.main()
