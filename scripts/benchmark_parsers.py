#!/usr/bin/env python3
"""Benchmark script for comparing PDB/CIF parsing libraries.

Compares:
- BioPython (Bio.PDB) - current implementation
- Biotite - fast Python library
- Gemmi - C++ library with Python bindings
- AtomWorks - Rosetta Commons framework (built on Biotite)
- FastPDB - Rust-based Biotite replacement

Usage:
    python scripts/benchmark_parsers.py
"""

import importlib.util
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# Test files
TEST_DIR = Path(__file__).parent.parent / "tests" / "data"
PDB_FILES = [
    TEST_DIR / "test_heavy_chain.pdb",
    TEST_DIR / "5omm_imgt.pdb",
    TEST_DIR / "12e8_imgt.pdb",
    TEST_DIR / "8sve_L.pdb",
]
CIF_FILES = [
    TEST_DIR / "test_minimal.cif",
]


def check_package(name: str) -> bool:
    """Check if a package is installed."""
    # Handle special case for biopython (package name != module name)
    if name == "biopython":
        return importlib.util.find_spec("Bio") is not None
    return importlib.util.find_spec(name) is not None


def timeit(func: Callable, n_iterations: int = 10) -> Tuple[float, float]:
    """Time a function over multiple iterations.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return statistics.mean(times), (
        statistics.stdev(times) if len(times) > 1 else 0.0
    )


class BiopythonBenchmark:
    """Benchmark for BioPython."""

    name = "BioPython"

    def __init__(self):
        from Bio import PDB

        self.PDB = PDB

    def read_pdb(self, filepath: Path) -> object:
        parser = self.PDB.PDBParser(QUIET=True)
        return parser.get_structure("structure", str(filepath))

    def read_cif(self, filepath: Path) -> object:
        parser = self.PDB.MMCIFParser(QUIET=True)
        return parser.get_structure("structure", str(filepath))

    def write_pdb(self, structure: object, filepath: Path) -> None:
        io = self.PDB.PDBIO()
        io.set_structure(structure)
        io.save(str(filepath))

    def write_cif(self, structure: object, filepath: Path) -> None:
        io = self.PDB.MMCIFIO()
        io.set_structure(structure)
        io.save(str(filepath))

    def extract_coords(self, structure: object) -> np.ndarray:
        """Extract backbone coordinates from structure."""
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0].strip():
                        continue
                    try:
                        n = residue["N"].get_coord()
                        ca = residue["CA"].get_coord()
                        c = residue["C"].get_coord()
                        coords.append([n, ca, c])
                    except KeyError:
                        continue
        return np.array(coords)


class BiotiteBenchmark:
    """Benchmark for Biotite."""

    name = "Biotite"

    def __init__(self):
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        import biotite.structure.io.pdbx as pdbx

        self.struc = struc
        self.pdb = pdb
        self.pdbx = pdbx

    def read_pdb(self, filepath: Path) -> object:
        pdb_file = self.pdb.PDBFile.read(str(filepath))
        return self.pdb.get_structure(pdb_file)

    def read_cif(self, filepath: Path) -> object:
        cif_file = self.pdbx.CIFFile.read(str(filepath))
        return self.pdbx.get_structure(cif_file)

    def write_pdb(self, structure: object, filepath: Path) -> None:
        pdb_file = self.pdb.PDBFile()
        self.pdb.set_structure(pdb_file, structure)
        pdb_file.write(str(filepath))

    def write_cif(self, structure: object, filepath: Path) -> None:
        cif_file = self.pdbx.CIFFile()
        self.pdbx.set_structure(cif_file, structure)
        cif_file.write(str(filepath))

    def extract_coords(self, structure: object) -> np.ndarray:
        """Extract backbone coordinates from structure."""
        # Get first model if it's a stack
        if hasattr(structure, "stack_depth"):
            structure = structure[0]

        # Filter for amino acids only
        aa_mask = self.struc.filter_amino_acids(structure)
        aa_atoms = structure[aa_mask]

        # Get backbone atoms (use filter_peptide_backbone in newer biotite)
        if hasattr(self.struc, "filter_peptide_backbone"):
            backbone_mask = self.struc.filter_peptide_backbone(aa_atoms)
        else:
            backbone_mask = self.struc.filter_backbone(aa_atoms)
        backbone = aa_atoms[backbone_mask]

        return backbone.coord


class GemmiBenchmark:
    """Benchmark for Gemmi."""

    name = "Gemmi"

    def __init__(self):
        import gemmi

        self.gemmi = gemmi

    def read_pdb(self, filepath: Path) -> object:
        return self.gemmi.read_structure(str(filepath))

    def read_cif(self, filepath: Path) -> object:
        return self.gemmi.read_structure(str(filepath))

    def write_pdb(self, structure: object, filepath: Path) -> None:
        structure.write_pdb(str(filepath))

    def write_cif(self, structure: object, filepath: Path) -> None:
        structure.make_mmcif_document().write_file(str(filepath))

    def extract_coords(self, structure: object) -> np.ndarray:
        """Extract backbone coordinates from structure."""
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.het_flag != "A":
                        continue
                    n = residue.find_atom("N", "*")
                    ca = residue.find_atom("CA", "*")
                    c = residue.find_atom("C", "*")
                    if n and ca and c:
                        coords.append(
                            [
                                [n.pos.x, n.pos.y, n.pos.z],
                                [ca.pos.x, ca.pos.y, ca.pos.z],
                                [c.pos.x, c.pos.y, c.pos.z],
                            ]
                        )
        return np.array(coords)


class FastPDBBenchmark:
    """Benchmark for FastPDB (Rust-based Biotite replacement)."""

    name = "FastPDB"

    def __init__(self):
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        import fastpdb

        self.fastpdb = fastpdb
        self.struc = struc
        self.pdbx = pdbx

    def read_pdb(self, filepath: Path) -> object:
        pdb_file = self.fastpdb.PDBFile.read(str(filepath))
        return pdb_file.get_structure()

    def read_cif(self, filepath: Path) -> object:
        # FastPDB only handles PDB files, use biotite for CIF
        cif_file = self.pdbx.CIFFile.read(str(filepath))
        return self.pdbx.get_structure(cif_file)

    def write_pdb(self, structure: object, filepath: Path) -> None:
        pdb_file = self.fastpdb.PDBFile()
        pdb_file.set_structure(structure)
        pdb_file.write(str(filepath))

    def write_cif(self, structure: object, filepath: Path) -> None:
        # FastPDB only handles PDB files, use biotite for CIF
        cif_file = self.pdbx.CIFFile()
        self.pdbx.set_structure(cif_file, structure)
        cif_file.write(str(filepath))

    def extract_coords(self, structure: object) -> np.ndarray:
        """Extract backbone coordinates from structure."""
        if hasattr(structure, "stack_depth"):
            structure = structure[0]
        aa_mask = self.struc.filter_amino_acids(structure)
        aa_atoms = structure[aa_mask]
        if hasattr(self.struc, "filter_peptide_backbone"):
            backbone_mask = self.struc.filter_peptide_backbone(aa_atoms)
        else:
            backbone_mask = self.struc.filter_backbone(aa_atoms)
        backbone = aa_atoms[backbone_mask]
        return backbone.coord


class AtomWorksBenchmark:
    """Benchmark for AtomWorks (Rosetta Commons)."""

    name = "AtomWorks"

    def __init__(self):
        from atomworks.io.parser import parse

        self.parse = parse

    def read_pdb(self, filepath: Path) -> object:
        return self.parse(str(filepath))

    def read_cif(self, filepath: Path) -> object:
        return self.parse(str(filepath))

    def write_pdb(self, structure: object, filepath: Path) -> None:
        # AtomWorks uses biotite's writing internally
        import biotite.structure.io.pdb as pdb

        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, structure)
        pdb_file.write(str(filepath))

    def write_cif(self, structure: object, filepath: Path) -> None:
        import biotite.structure.io.pdbx as pdbx

        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, structure)
        cif_file.write(str(filepath))

    def extract_coords(self, structure: object) -> np.ndarray:
        """Extract backbone coordinates from structure."""
        import biotite.structure as struc

        if hasattr(structure, "stack_depth"):
            structure = structure[0]
        aa_mask = struc.filter_amino_acids(structure)
        aa_atoms = structure[aa_mask]
        if hasattr(struc, "filter_peptide_backbone"):
            backbone_mask = struc.filter_peptide_backbone(aa_atoms)
        else:
            backbone_mask = struc.filter_backbone(aa_atoms)
        backbone = aa_atoms[backbone_mask]
        return backbone.coord


def run_benchmarks(
    benchmarks: List[object],
    pdb_files: List[Path],
    cif_files: List[Path],
    n_iterations: int = 10,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Run all benchmarks and return results."""
    results = {}

    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {benchmark.name}")
        print("=" * 60)

        results[benchmark.name] = {}

        # Benchmark PDB reading
        print("\n  PDB Reading:")
        pdb_read_times = []
        for pdb_file in pdb_files:
            if pdb_file.exists():
                bm = benchmark
                mean_time, std_time = timeit(
                    lambda f=pdb_file, b=bm: b.read_pdb(f), n_iterations
                )
                pdb_read_times.append(mean_time)
                print(f"    {pdb_file.name}: {mean_time:.2f} ms")

        if pdb_read_times:
            avg_read = statistics.mean(pdb_read_times)
            results[benchmark.name]["pdb_read"] = (
                avg_read,
                (
                    statistics.stdev(pdb_read_times)
                    if len(pdb_read_times) > 1
                    else 0
                ),
            )
            print(f"    Average: {avg_read:.2f} ms")

        # Benchmark CIF reading
        print("\n  CIF Reading:")
        cif_read_times = []
        for cif_file in cif_files:
            if cif_file.exists():
                bm = benchmark
                mean_time, std_time = timeit(
                    lambda f=cif_file, b=bm: b.read_cif(f), n_iterations
                )
                cif_read_times.append(mean_time)
                print(f"    {cif_file.name}: {mean_time:.2f} ms")

        if cif_read_times:
            avg_read = statistics.mean(cif_read_times)
            results[benchmark.name]["cif_read"] = (
                avg_read,
                (
                    statistics.stdev(cif_read_times)
                    if len(cif_read_times) > 1
                    else 0
                ),
            )
            print(f"    Average: {avg_read:.2f} ms")

        # Benchmark PDB writing
        print("\n  PDB Writing:")
        pdb_write_times = []
        for pdb_file in pdb_files:
            if pdb_file.exists():
                structure = benchmark.read_pdb(pdb_file)
                bm = benchmark
                with tempfile.NamedTemporaryFile(
                    suffix=".pdb", delete=True
                ) as tmp:
                    mean_time, std_time = timeit(
                        lambda s=structure, t=tmp.name, b=bm: b.write_pdb(
                            s, Path(t)
                        ),
                        n_iterations,
                    )
                    pdb_write_times.append(mean_time)
                    print(f"    {pdb_file.name}: {mean_time:.2f} ms")

        if pdb_write_times:
            avg_write = statistics.mean(pdb_write_times)
            results[benchmark.name]["pdb_write"] = (
                avg_write,
                (
                    statistics.stdev(pdb_write_times)
                    if len(pdb_write_times) > 1
                    else 0
                ),
            )
            print(f"    Average: {avg_write:.2f} ms")

        # Benchmark CIF writing
        print("\n  CIF Writing:")
        cif_write_times = []
        for pdb_file in pdb_files:  # Use PDB files as source, write to CIF
            if pdb_file.exists():
                structure = benchmark.read_pdb(pdb_file)
                bm = benchmark
                with tempfile.NamedTemporaryFile(
                    suffix=".cif", delete=True
                ) as tmp:
                    mean_time, std_time = timeit(
                        lambda s=structure, t=tmp.name, b=bm: b.write_cif(
                            s, Path(t)
                        ),
                        n_iterations,
                    )
                    cif_write_times.append(mean_time)
                    print(f"    {pdb_file.name} -> CIF: {mean_time:.2f} ms")

        if cif_write_times:
            avg_write = statistics.mean(cif_write_times)
            results[benchmark.name]["cif_write"] = (
                avg_write,
                (
                    statistics.stdev(cif_write_times)
                    if len(cif_write_times) > 1
                    else 0
                ),
            )
            print(f"    Average: {avg_write:.2f} ms")

        # Benchmark coordinate extraction
        print("\n  Coordinate Extraction:")
        coord_times = []
        for pdb_file in pdb_files:
            if pdb_file.exists():
                structure = benchmark.read_pdb(pdb_file)
                bm = benchmark
                mean_time, std_time = timeit(
                    lambda s=structure, b=bm: b.extract_coords(s),
                    n_iterations,
                )
                coord_times.append(mean_time)
                print(f"    {pdb_file.name}: {mean_time:.2f} ms")

        if coord_times:
            avg_extract = statistics.mean(coord_times)
            results[benchmark.name]["coord_extract"] = (
                avg_extract,
                statistics.stdev(coord_times) if len(coord_times) > 1 else 0,
            )
            print(f"    Average: {avg_extract:.2f} ms")

    return results


def print_summary(results: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    operations = [
        "pdb_read",
        "cif_read",
        "pdb_write",
        "cif_write",
        "coord_extract",
    ]
    op_names = [
        "PDB Read",
        "CIF Read",
        "PDB Write",
        "CIF Write",
        "Coord Extract",
    ]

    # Header
    print(f"\n{'Library':<15}", end="")
    for op_name in op_names:
        print(f"{op_name:>14}", end="")
    print(f"{'Total':>14}")
    print("-" * 85)

    # Data rows
    for lib_name, lib_results in results.items():
        print(f"{lib_name:<15}", end="")
        total = 0.0
        for op in operations:
            if op in lib_results:
                mean_val = lib_results[op][0]
                total += mean_val
                print(f"{mean_val:>12.2f}ms", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print(f"{total:>12.2f}ms")

    # Find fastest for each operation
    print("\n" + "-" * 85)
    print("Fastest library per operation:")
    for op, op_name in zip(operations, op_names):
        fastest_lib = None
        fastest_time = float("inf")
        for lib_name, lib_results in results.items():
            if op in lib_results and lib_results[op][0] < fastest_time:
                fastest_time = lib_results[op][0]
                fastest_lib = lib_name
        if fastest_lib:
            print(f"  {op_name}: {fastest_lib} ({fastest_time:.2f}ms)")

    # Calculate overall winner
    print("\n" + "-" * 85)
    totals = {}
    for lib_name, lib_results in results.items():
        total = sum(
            lib_results[op][0] for op in operations if op in lib_results
        )
        totals[lib_name] = total

    if totals:
        winner = min(totals, key=totals.get)
        print(f"OVERALL FASTEST: {winner} (total: {totals[winner]:.2f}ms)")


def main():
    """Run the benchmark suite."""
    print("=" * 80)
    print("PDB/CIF Parser Benchmark Suite")
    print("=" * 80)

    # Check available packages
    packages = {
        "BioPython": ("biopython", BiopythonBenchmark),
        "Biotite": ("biotite", BiotiteBenchmark),
        "Gemmi": ("gemmi", GemmiBenchmark),
        "FastPDB": ("fastpdb", FastPDBBenchmark),
        # AtomWorks excluded - does heavy processing (entity annotation)
        # that makes it unsuitable for raw parsing benchmarks
    }

    available_benchmarks = []

    print("\nChecking installed packages:")
    for name, (pkg, cls) in packages.items():
        if check_package(pkg):
            print(f"  [X] {name} ({pkg})")
            try:
                benchmark = cls()
                available_benchmarks.append(benchmark)
            except Exception as e:
                print(f"      Warning: Could not initialize {name}: {e}")
        else:
            print(f"  [ ] {name} ({pkg}) - not installed")

    if not available_benchmarks:
        print("\nNo benchmark libraries available!")
        sys.exit(1)

    # Check test files
    print("\nTest files:")
    for f in PDB_FILES + CIF_FILES:
        status = "exists" if f.exists() else "MISSING"
        print(f"  {f.name}: {status}")

    existing_pdb = [f for f in PDB_FILES if f.exists()]
    existing_cif = [f for f in CIF_FILES if f.exists()]

    if not existing_pdb and not existing_cif:
        print("\nNo test files found!")
        sys.exit(1)

    # Run benchmarks
    n_iterations = 20
    print(f"\nRunning benchmarks ({n_iterations} iterations per test)...")

    results = run_benchmarks(
        available_benchmarks, existing_pdb, existing_cif, n_iterations
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
