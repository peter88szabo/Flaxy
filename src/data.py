"""Read the extended-xyz scans and build the train / validation split."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import com_vector
from .units import ANG_BOHR, BOHR_ANG

_KV = re.compile(r"(\S+)=(\S+)")
_FILE = re.compile(r"R(?P<lo>[\d.]+)_to_(?P<hi>[\d.]+)_(?P<T>\d+)K")


def read_xyz(path, symbols, key="E_Bps"):
    """Return (names, X[n,atoms,3] in Angstrom, E[n] in Hartree); atom order must equal symbols."""
    lines = Path(path).read_text().splitlines()
    names, coords, energies = [], [], []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        header = lines[i + 1]
        fields = dict(_KV.findall(header))
        if fields.get("status", "PASS") != "PASS":
            raise ValueError(f"{path}: frame {header.split()[0]} has status {fields['status']}")
        atoms = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        if tuple(a[0] for a in atoms) != tuple(symbols):
            raise ValueError(f"{path}: unexpected atom order in {header.split()[0]}")
        names.append(header.split()[0])
        coords.append([[float(t) for t in a[1:4]] for a in atoms])
        energies.append(float(fields[key]))
        i += 2 + n
    return names, np.array(coords), np.array(energies)


def com_distance(X, system):
    """Distance between the fragment centres of mass, in the units of X."""
    return np.linalg.norm(com_vector(X, system), axis=1)


def _cluster(values, gap):
    """Label sorted 1-D values, starting a new cluster at every jump above gap."""
    order = np.argsort(values)
    labels = np.empty(len(values), dtype=int)
    labels[order] = np.concatenate([[0], np.cumsum(np.diff(values[order]) > gap)])
    return labels


@dataclass
class Dataset:
    names: np.ndarray       # frame names, unique within a file
    file: np.ndarray        # file index per geometry
    files: list             # file stems
    X: np.ndarray           # (n, 6, 3) bohr
    E: np.ndarray           # (n,) Hartree
    R: np.ndarray           # (n,) COM distance, Angstrom
    shell: np.ndarray       # (n,) index of the R shell within its file
    temperature: np.ndarray  # (n,) sampling temperature, K

    def __len__(self):
        return len(self.E)

    def subset(self, mask):
        return Dataset(self.names[mask], self.file[mask], self.files, self.X[mask],
                       self.E[mask], self.R[mask], self.shell[mask], self.temperature[mask])

    @property
    def keys(self):
        return np.array([f"{self.files[f]}:{n}" for f, n in zip(self.file, self.names)])


def load_dataset(data_dir, system, key="E_Bps"):
    """Read every *.xyz in data_dir. For two-fragment systems R is the COM distance
    (Angstrom) and `shell` clusters it; for one fragment R is NaN and shell 0. The
    sampling temperature is parsed from names like R1.5_to_4.5_300K (NaN otherwise)."""
    paths = sorted(Path(data_dir).glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"no .xyz files in {data_dir}")
    parts = []
    for f, path in enumerate(paths):
        names, X, E = read_xyz(path, system.symbols, key)
        if system.n_fragments == 2:
            R = com_distance(X, system)
            shell = _cluster(R, gap=0.05)
        else:
            R, shell = np.full(len(E), np.nan), np.zeros(len(E), dtype=int)
        match = _FILE.search(path.stem)
        T = float(match["T"]) if match else np.nan
        parts.append((np.array(names), np.full(len(E), f), X * ANG_BOHR, E, R, shell,
                      np.full(len(E), T)))
    cat = [np.concatenate(c) for c in zip(*parts)]
    return Dataset(cat[0], cat[1], [p.stem for p in paths], cat[2], cat[3], cat[4], cat[5], cat[6])


def make_split(ds, n_valid=300, seed=20261120):
    """Pick n_valid geometries per file, spread evenly over its R shells."""
    rng = np.random.default_rng(seed)
    split = {}
    for f, stem in enumerate(ds.files):
        idx = np.flatnonzero(ds.file == f)
        shells = np.unique(ds.shell[idx])
        quota = np.full(len(shells), n_valid // len(shells))
        quota[: n_valid - quota.sum()] += 1
        chosen = []
        for s, q in zip(shells, quota):
            members = idx[ds.shell[idx] == s]
            chosen.extend(rng.choice(members, size=min(q, len(members)), replace=False))
        split[stem] = sorted(ds.names[chosen].tolist())
    return split


def save_split(path, split, seed):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"seed": seed, "validation": split}, indent=1))


def load_split(path):
    return json.loads(Path(path).read_text())["validation"]


def validation_mask(ds, split):
    keys = {(stem, name) for stem, names in split.items() for name in names}
    return np.array([(ds.files[f], n) in keys for f, n in zip(ds.file, ds.names)])


def to_angstrom(X):
    return X * BOHR_ANG
