import os
import py_compile
import unittest

class TestCompile(unittest.TestCase):
    def test_vybn_recursive_emergence_compiles(self):
        path = os.path.join(
            'early_codex_experiments',
            'scripts',
            'cognitive_structures',
            'vybn_recursive_emergence.py',
        )
        py_compile.compile(path, doraise=True)

if __name__ == '__main__':
    unittest.main()
