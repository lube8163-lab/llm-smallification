#!/usr/bin/env python3
"""Verify and optionally archive RunPod Core ML endpoint packages."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path


DEFAULT_OUT_DIR = Path("/workspace/gemma12b/coreml-endpoints-seq4-int4")
NORM_PACKAGE = "gemma4_12b_norm_lm_head_1tok_int4_block32.mlpackage"
EMBEDDING_PACKAGE = "gemma4_12b_embedding_seq4_int4_block32.mlpackage"


def package_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def gib(size: int) -> str:
    return f"{size / 1024**3:.3f} GiB"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(packages: list[Path], manifest_path: Path) -> None:
    lines: list[str] = []
    for package in packages:
        for file_path in sorted(path for path in package.rglob("*") if path.is_file()):
            rel = file_path.relative_to(manifest_path.parent)
            lines.append(f"{file_digest(file_path)}  {rel}\n")
    manifest_path.write_text("".join(lines), encoding="utf-8")


def create_archive(packages: list[Path], manifest_path: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for package in packages:
            tar.add(package, arcname=package.name)
        tar.add(manifest_path, arcname=manifest_path.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--manifest", default="SHA256SUMS-endpoints")
    parser.add_argument("--tar-gz", help="optional .tar.gz archive path to create")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    required_names = [NORM_PACKAGE]
    if args.include_embedding:
        required_names.append(EMBEDDING_PACKAGE)

    packages: list[Path] = []
    errors: list[str] = []
    for name in required_names:
        package = out_dir / name
        if not package.is_dir():
            errors.append(f"missing {package}")
            continue
        packages.append(package)
        print(f"OK: {name} ({gib(package_size(package))})")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    manifest_path = out_dir / args.manifest
    write_manifest(packages, manifest_path)
    print(f"wrote {manifest_path}")

    if args.tar_gz:
        archive_path = Path(args.tar_gz)
        create_archive(packages, manifest_path, archive_path)
        print(f"wrote {archive_path} ({gib(archive_path.stat().st_size)})")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
