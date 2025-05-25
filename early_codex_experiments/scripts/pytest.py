import sys
import argparse
import unittest
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true')
    args, remaining = parser.parse_known_args()
    tests_path = os.path.join(os.path.dirname(__file__), '..', 'tests')
    unittest_args = [sys.argv[0], 'discover', '-s', tests_path] + remaining
    if args.quiet and '-q' not in unittest_args:
        unittest_args.append('-q')
    unittest.main(module=None, argv=unittest_args)


if __name__ == '__main__':
    main()
