"""Optional, isolated OpenMP setup for Apple Silicon; no writes outside this project."""

from pathlib import Path
import hashlib
import json
import platform
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise SystemExit(
        "Only Apple Silicon macOS needs this helper. Use your platform OpenMP runtime otherwise."
    )
meta = json.loads((ROOT / "libomp_provenance.json").read_text())
target = ROOT / ".venv/lib/libomp.dylib"
if not target.parent.exists():
    raise SystemExit("Create .venv and install requirements first.")
archive = ROOT / "llvm-openmp.tar.bz2"
if not archive.exists():
    urllib.request.urlretrieve(meta["source"], archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != meta["sha256"]:
    raise SystemExit("OpenMP package checksum mismatch; nothing installed.")
with tarfile.open(archive) as t:
    target.write_bytes(t.extractfile("lib/libomp.dylib").read())
    for name in t.getnames():
        if name.startswith("info/licenses/") and t.getmember(name).isfile():
            (target.parent / ("llvm-openmp-" + Path(name).name)).write_bytes(
                t.extractfile(name).read()
            )
print(
    "OpenMP installed within .venv; use python run.py for library-path configuration."
)
