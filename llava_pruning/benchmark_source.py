"""Freeze benchmark code and small source assets without copying weights/data."""

import hashlib
import json
from pathlib import Path


VERSION = 1
MANIFEST = ".source_manifest.json"
SOURCE_DIRS = ("llava_pruning", "vendor/llava/llava", "third_party", "configs")
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".yaml", ".yml", ".toml"}


def _hash(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def source_inventory(root):
    root = Path(root).resolve()
    paths = set(root.glob("*.py"))
    paths.update(p for p in root.iterdir() if p.is_file() and p.name.startswith(("LICENSE", "NOTICE")))
    for directory in SOURCE_DIRS:
        paths.update(p for p in (root / directory).rglob("*") if p.is_file()
                     and (p.suffix in TEXT_SUFFIXES or p.name.startswith(("LICENSE", "NOTICE"))))
    return {p.relative_to(root).as_posix(): _hash(p.read_bytes())
            for p in sorted(paths, key=lambda p: p.relative_to(root).as_posix())}


def _identity(files):
    entries = [name + ":" + files[name] for name in sorted(files)]
    return dict(version=VERSION, files=len(files), sha256=hashlib.sha256("\n".join(entries).encode()).hexdigest())


def freeze_source(project, destination):
    """Publish only a complete copy whose source stayed stable while copying."""
    project, destination = Path(project).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Source snapshot already exists: {destination}")
    files = source_inventory(project)
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True)
    for relative, expected in files.items():
        data = (project / relative).read_bytes()
        if _hash(data) != expected:
            raise ValueError("Source changed while freezing the experiment; restart with a new output directory")
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if source_inventory(project) != files or source_inventory(staging) != files:
        raise ValueError("Source changed while freezing the experiment; restart with a new output directory")
    (staging / MANIFEST).write_text(json.dumps(dict(version=VERSION, files=files), indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    return dict(root=str(destination), **_identity(files))


def verify_snapshot(snapshot, *, executing_root=None):
    root = Path(snapshot["root"]).resolve()
    if executing_root is not None and Path(executing_root).resolve() != root:
        raise ValueError(f"This experiment must execute its frozen worker: {root / 'benchmark_visa_methods.py'}")
    saved = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    identity = _identity(saved["files"])
    if saved["version"] != VERSION or any(snapshot.get(key) != value for key, value in identity.items()):
        raise ValueError("Frozen source manifest changed; refusing mixed experiment code")
    current = source_inventory(root)
    if current != saved["files"]:
        changed = sorted(name for name in set(current) | set(saved["files"])
                         if current.get(name) != saved["files"].get(name))
        raise ValueError("Frozen experiment source changed: " + ", ".join(changed[:5]))
    return root
