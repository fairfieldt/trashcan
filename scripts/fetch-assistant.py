#!/usr/bin/env python3
"""Fetch the pinned Gemma 4 12B assistant to ~/models/gemma-4-12b-assistant and check its sizes and hashes."""
import hashlib, json, pathlib, urllib.request
REVISION = "18934064dd4c5c6cc3621f6381e7d377fc8cb7bd"
EXPECTED = {
    "config.json": (2386, "7638c1d42f9fa73fe44b1a10604766b928b8985263f15775cd1a286a5a12799c"),
    "model.safetensors": (845719296, "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6"),
}
repo = 'google/gemma-4-12B-it-qat-q4_0-unquantized-assistant'
root = pathlib.Path.home() / 'models/gemma-4-12b-assistant'
root.mkdir(parents=True, exist_ok=True)
meta = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{REVISION}'))
revision = meta['sha']
assert revision == REVISION, 'unexpected revision'
manifest = {'repo': repo, 'revision': revision, 'files': {}}
for name in ['config.json', 'model.safetensors']:
    dst = root / name
    if not dst.exists():
        part = root / (name + '.partial')
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}') as src, part.open('wb') as out:
            while data := src.read(1024 * 1024): out.write(data)
        part.rename(dst)
    with dst.open('rb') as source: digest = hashlib.file_digest(source, 'sha256').hexdigest()
    assert (dst.stat().st_size, digest) == EXPECTED[name], f'checksum mismatch: {dst}'
    manifest['files'][name] = {'bytes': dst.stat().st_size, 'sha256': digest}
    print(name, manifest['files'][name], flush=True)
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('revision', revision)
