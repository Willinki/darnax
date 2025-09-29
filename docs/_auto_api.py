# docs/_auto_api.py
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from pathlib import PurePosixPath as Ppp
from typing import TYPE_CHECKING

from mkdocs_gen_files import open as gen_open

if TYPE_CHECKING:
    from collections.abc import Iterable

ROOT = "darnax"  # top-level package

# Make ./src importable without pip-install
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pkg_root = importlib.import_module(ROOT)


# ---------- Discovery ----------
def _is_public(name: str) -> bool:
    return not any(b in name for b in (".tests", "._", ".__"))


def _discover(root_pkg) -> list[str]:
    names = [root_pkg.__name__]
    for m in pkgutil.walk_packages(root_pkg.__path__, prefix=root_pkg.__name__ + "."):
        if m.ispkg or _is_public(m.name):
            names.append(m.name)
    # Parents first, stable order.
    return sorted(set(names), key=lambda s: (s.count("."), s))


MODS = _discover(pkg_root)


def _children_of(pkg: str) -> list[str]:
    pre, depth = pkg + ".", pkg.count(".") + 1
    return [m for m in MODS if m.startswith(pre) and m.count(".") == depth]


def _split_children(children: Iterable[str]) -> tuple[list[str], list[str]]:
    pkgs, mods = [], []
    for c in children:
        obj = importlib.import_module(c)
        (pkgs if hasattr(obj, "__path__") else mods).append(c)
    return pkgs, mods


def _rel_link(from_mod: str, to_mod: str) -> str:
    """Relative link from page for `from_mod` to page for `to_mod`."""
    frm_dir = Ppp(from_mod.replace(".", "/")).parent
    to_path = Ppp(to_mod.replace(".", "/"))
    return str(to_path.relative_to(frm_dir)) + ".md"


# ---------- Paths ----------
API_ROOT = Path("reference") / "api"  # relative to docs_dir
API_ROOT.mkdir(parents=True, exist_ok=True)


# ---------- Index (hierarchical, NOT flattened) ----------
root_children = _children_of(ROOT)
root_child_pkgs, root_child_mods = _split_children(root_children)

with gen_open(API_ROOT / "index.md", "w") as f:
    f.write("# API Index\n\n")
    f.write("Auto-generated. Top-level packages mirror the directory layout.\n\n")

    if root_child_pkgs:
        f.write("## Packages\n\n")
        for p in root_child_pkgs:
            # Link from index lives at reference/api/, so direct path is fine
            f.write(f"- [{p}]({p.replace('.', '/')}.md)")
            # First paragraph of package docstring as a teaser, if any
            doc = (importlib.import_module(p).__doc__ or "").strip()
            if doc:
                teaser = doc.split("\n\n", 1)[0].strip().replace("\n", " ")
                f.write(f" — {teaser}")
            f.write("\n")
        f.write("\n")

    if root_child_mods:
        f.write("## Modules\n\n")
        for m in root_child_mods:
            f.write(f"- [{m}]({m.replace('.', '/')}.md)\n")
        f.write("\n")


# ---------- Per-page generation ----------
for mod in MODS:
    page_path = API_ROOT / f"{mod.replace('.', '/')}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)

    obj = importlib.import_module(mod)
    is_pkg = hasattr(obj, "__path__")
    title = ROOT if mod == ROOT else mod

    with gen_open(page_path, "w") as f:
        f.write(f"# {title}\n\n")

        if is_pkg:
            # Hub page: no submodule rendering here; only structured links.
            if obj.__doc__:
                intro = obj.__doc__.strip().split("\n\n", 1)[0]
                if intro:
                    f.write(intro + "\n\n")

            kids = _children_of(mod)
            if kids:
                pkgs, leafs = _split_children(kids)

                if pkgs:
                    f.write("## Subpackages\n\n")
                    for p in pkgs:
                        f.write(f"- [{p}]({_rel_link(mod, p)})\n")
                    f.write("\n")

                if leafs:
                    f.write("## Modules\n\n")
                    for m in leafs:
                        f.write(f"- [{m}]({_rel_link(mod, m)})\n")
                    f.write("\n")
            else:
                f.write("_(Empty package)_\n")

        else:
            # Leaf module: render symbols here (single source of truth).
            f.write(f"::: {mod}\n")
            f.write("    options:\n")
            f.write("      show_root_full_path: false\n")
            f.write("      show_source: false\n")
            f.write("      members_order: source\n")
            f.write("      show_if_no_docstring: true\n")
            f.write("      filters: ['!^_', '!.+.__.*']\n")

print(f"[auto_api] Hierarchical index + {len(MODS)} pages written under reference/api/")
