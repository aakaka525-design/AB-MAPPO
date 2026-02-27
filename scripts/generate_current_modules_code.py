#!/usr/bin/env python3
"""
Generate docs/current_modules_code.md deterministically.

Collects all non-test Python modules in repo, sorted by relative path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXCLUDE_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "tests",
}


def _iter_python_modules(repo_root: Path):
    for path in sorted(repo_root.rglob("*.py"), key=lambda p: p.as_posix()):
        rel = path.relative_to(repo_root)
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            continue
        yield rel


def build_markdown(repo_root: Path) -> str:
    lines = [
        "# 当前代码模块源码汇总",
        "",
        "> 说明：该文档由脚本自动生成，收录当前仓库中非 tests 的 Python 模块完整源码。",
        "",
    ]
    for rel in _iter_python_modules(repo_root):
        src = (repo_root / rel).read_text(encoding="utf-8")
        lines.extend(
            [
                f"## {rel.as_posix()}",
                "",
                "```python",
                src.rstrip("\n"),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate docs/current_modules_code.md")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/current_modules_code.md"),
        help="Output markdown path relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(repo_root), encoding="utf-8")
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
