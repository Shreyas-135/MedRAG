#!/usr/bin/env python3
"""
import_registry_artifacts.py
=============================
Import a Kaggle-exported ``models/registry`` bundle into the local repository.

Supports two input formats:

  1. A **zip file** produced on Kaggle (e.g. ``registry_export.zip``) whose
     internal layout mirrors ``models/registry/``:

       models/registry/registry.json
       models/registry/checkpoints/<version>.pt
       ...

  2. A **folder** that *is* (or contains) the ``models/registry`` directory.

Usage
-----
  python scripts/import_registry_artifacts.py <source>

  <source>  Path to a zip file or a folder.

After import:
  - ``models/registry/registry.json`` is updated with the imported versions.
  - All ``*.pt`` checkpoint files are copied to ``models/registry/checkpoints/``.
  - All ``checkpoint_path`` values in the registry are rewritten to
    repo-relative paths using forward slashes (Windows-safe).

A final summary is printed showing the number of versions imported and how
many checkpoint files are missing after import.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# Locate repo root (this script lives in scripts/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = REPO_ROOT / 'models' / 'registry'
CHECKPOINT_DIR = REGISTRY_DIR / 'checkpoints'
REGISTRY_JSON = REGISTRY_DIR / 'registry.json'

# Kaggle prefix to strip from absolute checkpoint paths
KAGGLE_PREFIX = '/kaggle/working/MedRAG/'


def _strip_kaggle_prefix(path_str: str) -> str:
    """Convert an absolute Kaggle checkpoint path to a repo-relative path.

    Examples::

        /kaggle/working/MedRAG/models/registry/checkpoints/v1.pt
        → models/registry/checkpoints/v1.pt

    If the path does not start with the Kaggle prefix it is returned as-is
    (with path separators normalised to forward slashes).
    """
    # Normalise to forward slashes before any prefix work
    normalised = path_str.replace('\\', '/')
    if normalised.startswith(KAGGLE_PREFIX):
        return normalised[len(KAGGLE_PREFIX):]
    # Already relative or from a different absolute root – keep as-is
    return normalised


def _find_registry_root(folder: Path) -> Path:
    """Return the directory that directly contains ``registry.json``.

    Walks up to two levels below *folder* searching for ``registry.json``.
    Raises ``FileNotFoundError`` if not found.
    """
    for candidate in [
        folder,
        folder / 'registry',
        folder / 'models' / 'registry',
    ]:
        if (candidate / 'registry.json').is_file():
            return candidate

    # Broader recursive search (up to 4 levels deep)
    for p in folder.rglob('registry.json'):
        return p.parent

    raise FileNotFoundError(
        f"Could not locate registry.json anywhere inside '{folder}'. "
        "Make sure the archive/folder contains models/registry/registry.json."
    )


def import_from_folder(source_registry_dir: Path) -> None:
    """Import registry artifacts from *source_registry_dir* into the repo."""
    source_json = source_registry_dir / 'registry.json'
    if not source_json.is_file():
        print(f"ERROR: registry.json not found in '{source_registry_dir}'", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. Parse source registry                                            #
    # ------------------------------------------------------------------ #
    try:
        raw = source_json.read_text(encoding='utf-8').strip()
        if not raw:
            print("WARNING: Source registry.json is empty – nothing to import.")
            return
        source_data: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Source registry.json is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(source_data, dict):
        print("ERROR: Source registry.json must be a JSON object (dict).", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Ensure destination directories exist                             #
    # ------------------------------------------------------------------ #
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 3. Load existing local registry (merge, not overwrite)              #
    # ------------------------------------------------------------------ #
    existing_data: dict = {}
    if REGISTRY_JSON.is_file():
        try:
            existing_raw = REGISTRY_JSON.read_text(encoding='utf-8').strip()
            if existing_raw:
                existing_data = json.loads(existing_raw)
                if not isinstance(existing_data, dict):
                    existing_data = {}
        except Exception:
            existing_data = {}

    # ------------------------------------------------------------------ #
    # 4. Copy checkpoints and rewrite paths                               #
    # ------------------------------------------------------------------ #
    source_checkpoint_dir = source_registry_dir / 'checkpoints'
    imported = 0
    missing_after_import = []

    for version_id, entry in source_data.items():
        if not isinstance(entry, dict):
            print(f"  SKIP '{version_id}': entry is not a dict.")
            continue

        raw_path = entry.get('checkpoint_path', '')
        # Rewrite to repo-relative forward-slash path
        rel_path = _strip_kaggle_prefix(raw_path)

        # Derive checkpoint filename
        checkpoint_filename = Path(rel_path).name if rel_path else f"{version_id}.pt"

        # Try to copy the checkpoint file from the source bundle
        src_ckpt = source_checkpoint_dir / checkpoint_filename
        dst_ckpt = CHECKPOINT_DIR / checkpoint_filename

        if not dst_ckpt.exists():
            if src_ckpt.is_file():
                shutil.copy2(src_ckpt, dst_ckpt)
                print(f"  COPIED  {checkpoint_filename}")
            else:
                print(f"  MISSING {checkpoint_filename} (not in source bundle)")

        # Normalise path to repo-relative with forward slashes
        repo_rel = f"models/registry/checkpoints/{checkpoint_filename}"
        entry['checkpoint_path'] = repo_rel

        existing_data[version_id] = entry
        imported += 1

        if not dst_ckpt.exists():
            missing_after_import.append(checkpoint_filename)

    # ------------------------------------------------------------------ #
    # 5. Save merged registry                                             #
    # ------------------------------------------------------------------ #
    REGISTRY_JSON.write_text(json.dumps(existing_data, indent=2), encoding='utf-8')

    # ------------------------------------------------------------------ #
    # 6. Summary                                                          #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 55)
    print(f"  Import complete")
    print(f"  Versions imported : {imported}")
    print(f"  Missing checkpoints: {len(missing_after_import)}")
    if missing_after_import:
        print("  Missing files:")
        for name in missing_after_import:
            print(f"    - {name}")
    print(f"  Registry saved to : {REGISTRY_JSON}")
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import Kaggle-trained model registry artifacts into the local repo.\n"
            "Accepts a zip file or a folder containing models/registry/."
        )
    )
    parser.add_argument(
        'source',
        help="Path to a zip file or folder containing models/registry/",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: '{source}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if source.is_file() and source.suffix.lower() == '.zip':
        # Extract to a temporary directory, then import
        with tempfile.TemporaryDirectory(prefix='medrag_import_') as tmpdir:
            print(f"Extracting '{source.name}' …")
            with zipfile.ZipFile(source, 'r') as zf:
                zf.extractall(tmpdir)
            extracted = Path(tmpdir)
            try:
                registry_root = _find_registry_root(extracted)
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"Found registry at: {registry_root.relative_to(extracted)}")
            import_from_folder(registry_root)
    elif source.is_dir():
        try:
            registry_root = _find_registry_root(source)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Importing from folder: {registry_root}")
        import_from_folder(registry_root)
    else:
        print(
            f"ERROR: '{source}' is not a zip file or a directory.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
