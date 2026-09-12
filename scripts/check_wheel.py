"""Build-independent wheel installation smoke test; downloads wheel dependencies."""

from pathlib import Path
import os
import subprocess
import tempfile
import venv


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            "Expected exactly one wheel in dist; build in a clean checkout."
        )
    with tempfile.TemporaryDirectory(prefix="memory-wheel-") as directory:
        work = Path(directory)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", f"{wheels[0]}[mcp]"],
            check=True,
            cwd=work,
        )
        code = """
import json
from pathlib import Path
import sys
import agent_memory
from agent_memory import MemoryStore, HashingEmbedder
from agent_memory.mcp_server import build_server
assert Path(sys.prefix) in Path(agent_memory.__file__).parents
store = MemoryStore('smoke.json', embedder=HashingEmbedder())
entry = store.write('Legacy identity stays usable.', id='mem_0001')
store.update(entry.id, text='Corrected after install.', expected_revision=1)
assert MemoryStore('smoke.json', embedder=HashingEmbedder()).get(entry.id).revision == 2
assert build_server(Path('mcp.json')) is not None
print('Installed wheel smoke passed:', agent_memory.__version__)
"""
        env = dict(os.environ, AGENT_MEMORY_EMBEDDER="hashing")
        env.pop("PYTHONPATH", None)
        subprocess.run([str(python), "-I", "-c", code], check=True, cwd=work, env=env)
        command = bin_dir / ("agent-memory.exe" if os.name == "nt" else "agent-memory")
        subprocess.run(
            [str(command), "--path", str(work / "smoke.json"), "doctor", "--json"],
            check=True,
            cwd=work,
            env=env,
        )


if __name__ == "__main__":
    main()
