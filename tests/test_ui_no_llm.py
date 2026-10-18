"""The Dashboard code path never imports an LLM provider (or the agent / categorizer that can build one).

1. Static: the modules on the dashboard path have no module-level import of model code.
2. Dynamic: in a fresh process where importing model code raises, app.py renders the full dashboard
   for two months without an exception.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = ["app.py", "src/ui/__init__.py", "src/ui/dashboard.py", "src/ui/components.py", "src/ui/theme.py",
                  "src/ui/format.py", "src/ui/charts.py", "src/ui/data.py", "src/ui/sidebar.py"]
FORBIDDEN = ("src.llm", "src.agent", "src.categorizer", "src.advisor", "ollama", "langchain")


def _module_level_imports(path: Path) -> list[str]:
    """Imports executed when the module is imported (function bodies excluded: those are lazy)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    return names


def test_dashboard_modules_import_no_model_code():
    for rel in DASHBOARD_PATH:
        bad = [n for n in _module_level_imports(ROOT / rel) if n.startswith(FORBIDDEN)]
        assert not bad, f"{rel} imports {bad}"


BLOCKED_RUN = textwrap.dedent("""
    import sys
    BLOCKED = {blocked!r}

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.startswith(BLOCKED):
                raise ImportError(f"dashboard path must not import {{name}}")
            return None

    sys.meta_path.insert(0, Block())
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["db_path"] = {db!r}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert sum('class="fa-kpi"' in m.value for m in at.markdown) == 4
    at.selectbox(key="month").set_value("2026-03").run()
    assert not at.exception, [e.value for e in at.exception]
    loaded = [m for m in sys.modules if m.startswith(BLOCKED)]
    assert not loaded, loaded
    print("OK")
""")


def test_dashboard_renders_with_model_code_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_APP_DATA", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    from src.ui.data import ensure_sample_db
    db = ensure_sample_db()  # built here: the build itself uses the categorizer
    script = BLOCKED_RUN.format(blocked=FORBIDDEN, db=str(db))
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr
