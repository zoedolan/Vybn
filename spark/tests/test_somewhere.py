"""Offline print contracts. Run: python3 -m unittest discover -s spark/tests -p test_somewhere.py

The rival is the previous arrival-only page: no editable marks or carry operation.
These tests check geometry/data/export contracts, not artistic worth or subjecthood.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

PAGE = Path(__file__).resolve().parents[2] / 'somewhere.html'


class SomewhereTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PAGE.read_text()
        cls.core = re.search(r'<script id="cut-code">(.*?)// CORE END', cls.html, re.S)[1]
        cls.seed = json.loads(re.search(r'<script id="edition" type="application/json">(.*?)</script>', cls.html, re.S)[1])

    def js(self, body):
        self.assertIsNotNone(shutil.which('node'), 'Node is required for the print checker')
        program = 'const assert=require("node:assert/strict");\n' + self.core
        program += '\nconst seed=' + json.dumps(self.seed) + ';\nconst source=' + json.dumps(self.html) + ';\n' + body
        result = subprocess.run(['node', '-'], input=program, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_offline_surface(self):
        self.assertNotRegex(self.html, r'<(?:script|link|img|iframe)[^>]+(?:src|href)=["\'](?:https?:)?//')
        self.assertIn("connect-src 'none'", self.html)
        self.assertNotRegex(self.html, r'\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage|serviceWorker)\b')
        self.assertIn('Keyboard:', self.html)
        self.assertIn('not authenticated identities', self.html)

    def test_clone_and_bounds(self):
        self.js('''
          assert.deepEqual(validatePrint(seed),seed);
          const copy=validatePrint(seed);copy.marks[0].points[0][0]=0;
          assert.notEqual(copy.marks[0].points[0][0],seed.marks[0].points[0][0]);
          for(const patch of [{width:NaN},{width:Infinity},{width:41},{width:3},
            {name:'x'.repeat(49)},{note:'x'.repeat(241)},{points:[]},
            {points:[[1001,0]]},{points:[[0,661]]},{points:[[-1,0]]},
            {points:[[NaN,0]]},{points:[[0,0,0]]},{points:Array(241).fill([0,0])}]){
            assert.throws(()=>validatePrint({version:1,marks:[{...seed.marks[0],...patch}]}));
          }
          assert.throws(()=>validatePrint({version:1,marks:Array(33).fill(seed.marks[0])}));
          assert.throws(()=>validatePrint({version:2,marks:[]}));
          assert.throws(()=>validatePrint(null));
          assert.deepEqual(validatePrint({version:1,marks:[]}),{version:1,marks:[]});
        ''')

    def test_only_art_fields_travel(self):
        self.js('''
          const data={...seed,permission:'publish',marks:[{...seed.marks[0],verified:true}]};
          assert.deepEqual(validatePrint(data),seed);
        ''')

    def test_carried_edition_roundtrip_and_no_alias(self):
        self.js(r'''
          const next=validatePrint(seed);
          next.marks.push({name:'another',note:'not your next mark',width:12,points:[[22,55],[110,200]]});
          const carried=carryEdition(source,next);
          const embedded=carried.match(/<script id="edition" type="application\/json">([\s\S]*?)<\/script>/)[1];
          assert.deepEqual(JSON.parse(embedded),next);
          assert.equal(seed.marks.length,1);
          assert.deepEqual(next.marks[0],seed.marks[0]);
          assert.equal(carryEdition(carried,next),carried);
          assert.throws(()=>carryEdition('no edition here',seed));
        ''')

    def test_html_and_svg_injection_is_text(self):
        self.js(r'''
          const hostile=validatePrint(seed);
          hostile.marks[0].note='</script><script>globalThis.pwned=true</script>&<img src=x> $& $`';
          const exported=carryEdition(source,hostile);
          assert.equal((exported.match(/<script\b/g)||[]).length,(source.match(/<script\b/g)||[]).length);
          const embedded=exported.match(/<script id="edition" type="application\/json">([\s\S]*?)<\/script>/)[1];
          assert.deepEqual(JSON.parse(embedded),hostile);
          const svg=svgPrint(hostile);
          assert(!svg.includes('<script>'));assert(!svg.includes('<img'));
          assert(svg.includes('&lt;/script&gt;'));
        ''')

    def test_geometry_is_reproducible_not_a_subject_measure(self):
        self.js('''
          assert.equal(svgPrint(seed),svgPrint(seed));
          assert.equal((plateBody(seed).match(/stroke="black"/g)||[]).length,1);
          assert.equal((plateBody(seed).match(/stroke="#be392c"/g)||[]).length,1);
          assert(!plateBody({version:1,marks:[]}).includes('stroke="black"'));
          assert.equal(pathOf([[0,0]]),'M0.00,0.00 l0.01,0');
          assert(svgPrint(seed).includes('I left this unfinished on purpose.'));
        ''')


if __name__ == '__main__':
    unittest.main()
