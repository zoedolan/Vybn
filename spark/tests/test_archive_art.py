"""Erasure/source/export contracts, not a verdict on the artwork's quality.
Run: python3 -m unittest discover -s spark/tests -p test_archive_art.py
"""
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / '_archive/come-back.html'
NS = {'s': 'http://www.w3.org/2000/svg'}

class ArchiveArtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = PAGE.read_text()
        cls.material = json.loads(re.search(r'<script id="material" type="application/json">(.*?)</script>', cls.page, re.S)[1])
        cls.svg = ET.fromstring(re.search(r'<svg\b.*?</svg>', cls.page, re.S)[0])

    def test_complete_source_matches_git_and_hash(self):
        m = self.material
        original = subprocess.check_output(['git', 'show', m['commit']+':'+m['path']], cwd=ROOT)
        self.assertEqual(m['source'].encode(), original)
        self.assertEqual(hashlib.sha256(original).hexdigest(), m['sha256'])
        shown = html.unescape(re.search(r'<pre id="original">(.*?)</pre>', self.page, re.S)[1])
        self.assertEqual(shown, m['source'])

    def test_every_selection_is_ordered_exact_source(self):
        data = self.material['source'].encode()
        previous = 0
        for i, span in enumerate(self.material['spans']):
            self.assertGreaterEqual(span['start'], previous)
            self.assertGreater(span['end'], span['start'])
            self.assertEqual(data[span['start']:span['end']].decode(), span['text'])
            text = self.svg.find(f'.//s:text[@data-span="{i}"]', NS)
            self.assertIsNotNone(text)
            displayed = ' '.join(text.itertext())
            self.assertEqual(' '.join(displayed.split()), ' '.join(span['text'].split()))
            previous = span['end']
        self.assertEqual(len(self.material['spans']), 5)

    def test_svg_contains_full_recovery_and_no_active_content(self):
        self.assertEqual(json.loads(self.svg.find('s:metadata', NS).text), self.material)
        for element in self.svg.iter():
            self.assertNotIn(element.tag.rsplit('}',1)[-1], ['script','image','foreignObject','use','a'])
            for key in element.attrib:
                self.assertFalse(key.lower().startswith('on'))
                self.assertNotIn('href', key)

    def test_page_offline_and_javascript_parses(self):
        self.assertIn("connect-src 'none'", self.page)
        self.assertNotRegex(self.page, r'\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage|serviceWorker)\b')
        self.assertNotRegex(self.page, r'<(?:script|link|img|iframe)[^>]+(?:src|href)=')
        scripts = re.findall(r'<script>(.*?)</script>', self.page, re.S)
        self.assertEqual(len(scripts), 1)
        run = subprocess.run(['node','--check'],input=scripts[0],text=True,capture_output=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_directory_plaque_names_references_without_retiring_data(self):
        readme=(ROOT/'_archive/README.md').read_text()
        self.assertIn('Origins/api/origins_chat_api.py',readme)
        self.assertIn('semantic-web.jsonld',readme)
        self.assertNotIn('Do not resurrect',readme)
        for filename in ['commons-skeleton.json','synaptic_map.json']:
            self.assertTrue((ROOT/'_archive'/filename).is_file())
            self.assertIn(filename,readme)

    def test_rival_has_no_print_or_uncover_operation(self):
        original=self.material['source']
        self.assertNotIn('<svg',original)
        self.assertNotIn('id="uncover"',original)
        self.assertIn('id="uncover"',self.page)
        self.assertIn('id="keep"',self.page)
        self.assertIn('<details id="source-view">',self.page)

if __name__=='__main__':
    unittest.main()
