"""Exercise checkpoint transfer with a tiny fixture, NOT a trained Hindi model."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    full = json.loads((here/'manifests/modi.json').read_text())
    # Check the prepared real dataset for filename-family and pixel leakage.
    for key in ('group', 'digest'):
        assignments = {}
        for row in full['records']:
            previous = assignments.setdefault(row[key], row['split'])
            assert previous == row['split'], f'{key} leaks across splits'
    with tempfile.TemporaryDirectory(prefix='modi-model-smoke-') as temp:
        root = Path(temp)
        def fixture(name, kind, count):
            labels = full['labels'][:count]
            records = []
            for label in labels:
                for split in ('train', 'val', 'test'):
                    matches = [r for r in full['records'] if r['label'] == label and r['split'] == split]
                    records.extend(matches[:2 if split == 'train' else 1])
            path = root/f'{name}.json'
            path.write_text(json.dumps(dict(root=full['root'], kind=kind, labels=labels, records=records)))
            return path
        # 'devanagari' metadata below is ONLY to test the source checkpoint contract.
        source = fixture('source-fixture', 'devanagari', 2)
        target = fixture('target-fixture', 'modi', 3)
        def run(*args):
            subprocess.run([sys.executable, str(here/'train.py'), *map(str,args)], check=True)
        run('--manifest', source, '--out', root/'source', '--epochs', 1, '--device', 'cpu', '--batch-size', 4)
        run('--manifest', target, '--out', root/'target', '--source', root/'source/best.pt',
            '--epochs', 2, '--warmup', 1, '--device', 'cpu', '--batch-size', 6)
        run('--manifest', target, '--out', root/'target', '--evaluate', root/'target/best.pt', '--device', 'cpu')
        from predict import Recognizer
        from PIL import Image
        with Image.open(Path(full['root'])/full['records'][0]['path']) as image:
            prediction = Recognizer(root/'target/best.pt').predict(image)
        assert len(prediction) == 3
        assert {p['label'] for p in prediction} == set(full['labels'][:3])
    print('PASS: real-data split isolation, source training, head replacement, frozen/unfrozen training, evaluation, inference.')
    print('Temporary smoke checkpoints removed. No Hindi/Modi accuracy claim is made by this test.')


if __name__ == '__main__':
    main()
