"""Regression checks for official-test duplicate handling."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from prepare import prepare


class PrepareTests(unittest.TestCase):
    def test_test_overlap_excludes_entire_training_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            for split in ('Train', 'Test'):
                (root / split / 'ka').mkdir(parents=True)
            def image(path, value):
                Image.new('L', (8, 8), value).save(root / path)
            image('Test/ka/heldout.png', 25)
            image('Train/ka/copy_original.png', 25)
            image('Train/ka/copy_aug_1.png', 26)
            for i in range(6):
                image(f'Train/ka/source{i}.png', 100+i)
            output = Path(directory) / 'manifest.json'
            report = prepare(root, output, 'devanagari')
            data = json.loads(output.read_text())
            self.assertEqual(report['excluded_training_overlap_images'], 2)
            self.assertEqual(report['official_overlap_groups'], 1)
            self.assertEqual([r['path'] for r in data['records'] if r['split'] == 'test'],
                             ['Test/ka/heldout.png'])
            self.assertFalse(any('copy_' in r['path'] for r in data['records']))
            self.assertEqual(len(data['excluded_records']), 2)
            for key in ('digest', 'group'):
                assignments = {}
                for row in data['records']:
                    self.assertEqual(assignments.setdefault(row[key], row['split']), row['split'])


if __name__ == '__main__':
    unittest.main()
