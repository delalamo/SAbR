#!/usr/bin/env python3
"""Optimized OAS MPNN embedding generation with multi-noise support.

This script generates averaged MPNN embeddings from ~148,832 OAS antibody
structure models across multiple noise levels. Key optimizations:

1. Length-sorted processing: Sorts structures by (heavy_len, light_len) to
   minimize JAX recompilation triggered by varying sequence lengths.

2. Multi-noise efficiency: Loads each PDB once and computes embeddings for
   all 5 noise levels (0.0, 0.2, 0.5, 1.0, 2.0 angstroms).

3. External noise application: Applies noise before model call, enabling
   a single compiled model for all noise levels.

4. Robust checkpointing: Saves state every 1000 structures for resume
   capability, plus intermediate results every 10000 structures.

Output format matches sabr/assets/embeddings.npz:
    Per noise level directory:
        {chain_type}.npz with:
            'name': str,           # 'heavy', 'kappa', or 'lambda'
            'array': float32,      # shape (N, 64) - averaged embeddings
            'idxs': str array,     # shape (N,) - IMGT position strings

Usage:
    # Test on small subset
    python scripts/generate_oas_embeddings_optimized.py \\
        --limit 100 --output-dir test_output

    # Full run
    python scripts/generate_oas_embeddings_optimized.py \\
        --output-dir oas_embeddings_optimized

    # Resume from checkpoint
    python scripts/generate_oas_embeddings_optimized.py \\
        --output-dir oas_embeddings_optimized
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# Add sabr to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sabr import constants  # noqa: E402
from sabr.embeddings.backend import (  # noqa: E402
    _convert_numpy_to_jax,
    _unflatten_dict,
)
from sabr.embeddings.inputs import get_inputs  # noqa: E402
from sabr.nn.end_to_end import END_TO_END  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable


DEFAULT_PDB_DIR = "OAS_models/structures"
DEFAULT_CSV_PATH = "OAS_models/OAS_paired_all.csv"
DEFAULT_OUTPUT_DIR = "oas_embeddings_optimized"
DEFAULT_NOISE_LEVELS = [0.0, 0.2, 0.5, 1.0, 2.0]
RESIDUE_RANGE = (1, 128)
CHECKPOINT_INTERVAL = 1000
SAVE_INTERVAL = 10000
MIN_PRESENCE_FRACTION = 0.33

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_mpnn_params(params_path: str = "sabr.assets") -> dict:
    """Load MPNN encoder parameters from package resources."""
    package_files = files(params_path)
    npz_path = package_files / "mpnn_encoder.npz"

    with open(npz_path, "rb") as f:
        data = dict(np.load(f, allow_pickle=False))

    params = _unflatten_dict(data)
    params = _convert_numpy_to_jax(params)
    return params


def create_e2e_model_no_noise() -> END_TO_END:
    """Create END_TO_END model with zero noise (noise applied externally)."""
    return END_TO_END(
        constants.EMBED_DIM,
        constants.EMBED_DIM,
        constants.EMBED_DIM,
        constants.N_MPNN_LAYERS,
        constants.EMBED_DIM,
        affine=True,
        soft_max=False,
        dropout=0.0,
        augment_eps=0.0,
    )


def _compute_embeddings_fn(
    coords: np.ndarray,
    mask: np.ndarray,
    chain_ids: np.ndarray,
    residue_indices: np.ndarray,
) -> np.ndarray:
    """Compute MPNN embeddings from structure coordinates."""
    model = create_e2e_model_no_noise()
    return model.MPNN(coords, mask, chain_ids, residue_indices)


class MultiNoiseEmbeddingBackend:
    """Backend for computing embeddings at multiple noise levels efficiently."""

    def __init__(self, seed: int = 42):
        """Initialize backend with model parameters.

        Args:
            seed: Base random seed for reproducibility.
        """
        self.params = load_mpnn_params()
        self._transformed_fn = hk.transform(_compute_embeddings_fn)
        self.base_seed = seed
        LOGGER.info("Initialized MultiNoiseEmbeddingBackend")
        LOGGER.info(f"JAX devices: {jax.devices()}")

    def compute_all_noise_levels(
        self,
        coords: np.ndarray,
        mask: np.ndarray,
        chain_ids: np.ndarray,
        residue_indices: np.ndarray,
        noise_levels: list,
        structure_idx: int,
    ) -> dict:
        """Compute embeddings for all noise levels from single coord load.

        Args:
            coords: Backbone coordinates [1, N, 4, 3].
            mask: Binary mask for valid residues [1, N].
            chain_ids: Chain identifiers [1, N].
            residue_indices: Sequential residue indices [1, N].
            noise_levels: List of noise standard deviations to apply.
            structure_idx: Index used for deterministic random seed.

        Returns:
            Dictionary mapping noise level to embedding array [N, 64].
        """
        results = {}
        coords_jax = jnp.array(coords)

        for noise_idx, noise in enumerate(noise_levels):
            seed_val = (self.base_seed + structure_idx * 100 + noise_idx) % (
                2**31 - 1
            )
            key = jax.random.PRNGKey(seed_val)

            if noise > 0:
                noisy_coords = coords_jax + noise * jax.random.normal(
                    key, coords_jax.shape
                )
            else:
                noisy_coords = coords_jax

            result = self._transformed_fn.apply(
                self.params,
                key,
                noisy_coords,
                mask,
                chain_ids,
                residue_indices,
            )
            results[noise] = np.asarray(result[0])

        return results


def load_sorted_structures(csv_path: str, cache_path: str = None) -> list:
    """Load structures sorted by (heavy_len, light_len).

    Sorting minimizes JAX recompilation by grouping similar-length sequences.

    Args:
        csv_path: Path to OAS_paired_all.csv.
        cache_path: Optional path to cache sorted structure list.

    Returns:
        List of structure dicts sorted by (heavy_len, light_len).
    """
    if cache_path and Path(cache_path).exists():
        LOGGER.info(f"Loading cached structure list from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    LOGGER.info(f"Loading and sorting structures from {csv_path}")
    structures = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["ID"]
            full_seq = row["full_seq"]
            heavy_seq, light_seq = full_seq.split("/")
            locus_light = row["locus_light"]

            structures.append(
                {
                    "id": pdb_id,
                    "heavy_len": len(heavy_seq),
                    "light_len": len(light_seq),
                    "locus": locus_light,
                }
            )

    structures = sorted(
        structures, key=lambda x: (x["heavy_len"], x["light_len"])
    )
    LOGGER.info(f"Loaded {len(structures)} structures")

    if cache_path:
        LOGGER.info(f"Caching sorted structure list to {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(structures, f)

    return structures


def is_valid_residue_id(residue_id: str) -> bool:
    """Check if residue ID is valid (no insertion code, in range 1-128)."""
    if not residue_id.isdigit():
        return False
    resnum = int(residue_id)
    return RESIDUE_RANGE[0] <= resnum <= RESIDUE_RANGE[1]


def create_accumulators(noise_levels: list) -> dict:
    """Create nested accumulators for position-wise embedding sums.

    Returns:
        Dict: chain -> noise -> position -> {'sum': array, 'count': int}
    """
    return {
        chain: {
            noise: defaultdict(
                lambda: {
                    "sum": np.zeros(constants.EMBED_DIM, dtype=np.float64),
                    "count": 0,
                }
            )
            for noise in noise_levels
        }
        for chain in ["heavy", "kappa", "lambda"]
    }


def update_accumulator(
    accumulators: dict,
    chain_type: str,
    noise: float,
    residue_ids: list,
    embeddings: np.ndarray,
) -> None:
    """Update running sum/count for each IMGT position."""
    for i, res_id in enumerate(residue_ids):
        if is_valid_residue_id(res_id):
            accumulators[chain_type][noise][res_id]["sum"] += embeddings[
                i
            ].astype(np.float64)
            accumulators[chain_type][noise][res_id]["count"] += 1


def accumulators_to_serializable(accumulators: dict) -> dict:
    """Convert accumulators to JSON-serializable format."""
    result = {}
    for chain in accumulators:
        result[chain] = {}
        for noise in accumulators[chain]:
            noise_key = str(noise)
            result[chain][noise_key] = {}
            for pos, data in accumulators[chain][noise].items():
                result[chain][noise_key][pos] = {
                    "sum": data["sum"].tolist(),
                    "count": data["count"],
                }
    return result


def accumulators_from_serializable(data: dict, noise_levels: list) -> dict:
    """Restore accumulators from JSON-serializable format."""
    accumulators = create_accumulators(noise_levels)
    for chain in data:
        for noise_key in data[chain]:
            noise = float(noise_key)
            for pos, pos_data in data[chain][noise_key].items():
                accumulators[chain][noise][pos]["sum"] = np.array(
                    pos_data["sum"], dtype=np.float64
                )
                accumulators[chain][noise][pos]["count"] = pos_data["count"]
    return accumulators


def save_checkpoint(
    checkpoint_path: Path, accumulators: dict, last_idx: int
) -> None:
    """Save checkpoint for resume capability."""
    checkpoint = {
        "last_idx": last_idx,
        "accumulators": accumulators_to_serializable(accumulators),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f)
    tmp_path.rename(checkpoint_path)
    LOGGER.info(f"Saved checkpoint at index {last_idx}")


def load_checkpoint(checkpoint_path: Path, noise_levels: list) -> tuple:
    """Load checkpoint if exists.

    Returns:
        Tuple of (accumulators, start_idx) or (None, 0) if no checkpoint.
    """
    if not checkpoint_path.exists():
        return None, 0

    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    accumulators = accumulators_from_serializable(
        checkpoint["accumulators"], noise_levels
    )
    start_idx = checkpoint["last_idx"] + 1
    LOGGER.info(
        f"Resuming from checkpoint at index {start_idx} "
        f"(saved {checkpoint['timestamp']})"
    )
    return accumulators, start_idx


def save_embeddings_to_dir(
    output_dir: Path,
    accumulators: dict,
    noise_levels: list,
    chain_types: list,
) -> None:
    """Save averaged embeddings to NPZ files organized by noise level."""
    for noise in noise_levels:
        noise_dir = output_dir / f"noise_{noise}"
        noise_dir.mkdir(parents=True, exist_ok=True)

        for chain_type in chain_types:
            pos_data = accumulators[chain_type][noise]

            valid_positions = sorted(
                [pos for pos, data in pos_data.items() if data["count"] > 0],
                key=lambda x: int(x),
            )

            if not valid_positions:
                LOGGER.warning(
                    f"No valid positions for {chain_type} at noise {noise}"
                )
                continue

            array = np.zeros(
                (len(valid_positions), constants.EMBED_DIM), dtype=np.float32
            )
            for i, pos in enumerate(valid_positions):
                array[i] = (
                    pos_data[pos]["sum"] / pos_data[pos]["count"]
                ).astype(np.float32)

            idxs = np.array(valid_positions, dtype="<U3")

            output_path = noise_dir / f"{chain_type}.npz"
            np.savez(
                output_path,
                name=chain_type,
                array=array,
                idxs=idxs,
            )
            LOGGER.info(
                f"Saved {chain_type} at noise {noise}: "
                f"shape {array.shape}, positions {idxs[0]}-{idxs[-1]}"
            )


def save_intermediate_embeddings(
    output_dir: Path,
    accumulators: dict,
    noise_levels: list,
    n_structures: int,
) -> None:
    """Save current averaged embeddings as intermediate usable results."""
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)

    for noise in noise_levels:
        noise_dir = intermediate_dir / f"noise_{noise}_n{n_structures}"
        noise_dir.mkdir(exist_ok=True)

        for chain_type in ["heavy", "kappa", "lambda"]:
            pos_data = accumulators[chain_type][noise]

            valid_positions = sorted(
                [pos for pos, data in pos_data.items() if data["count"] > 0],
                key=lambda x: int(x),
            )

            if not valid_positions:
                continue

            array = np.zeros(
                (len(valid_positions), constants.EMBED_DIM), dtype=np.float32
            )
            for i, pos in enumerate(valid_positions):
                array[i] = (
                    pos_data[pos]["sum"] / pos_data[pos]["count"]
                ).astype(np.float32)

            idxs = np.array(valid_positions, dtype="<U3")

            np.savez(
                noise_dir / f"{chain_type}.npz",
                name=chain_type,
                array=array,
                idxs=idxs,
            )


def process_structures(
    pdb_dir: str,
    csv_path: str,
    output_dir: str,
    noise_levels: list,
    limit: int = None,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
) -> None:
    """Process all structures and generate averaged embeddings.

    Args:
        pdb_dir: Directory containing PDB files.
        csv_path: Path to OAS metadata CSV.
        output_dir: Output directory for results.
        noise_levels: List of noise levels to process.
        limit: Optional limit on structures to process.
        checkpoint_interval: Save checkpoint every N structures.
        save_interval: Save intermediate results every N structures.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = output_path / "lengths_cache.json"
    structures = load_sorted_structures(csv_path, str(cache_path))

    if limit:
        structures = structures[:limit]
        LOGGER.info(f"Limited to {limit} structures")

    checkpoint_path = output_path / "checkpoint.json"
    accumulators, start_idx = load_checkpoint(checkpoint_path, noise_levels)

    if accumulators is None:
        accumulators = create_accumulators(noise_levels)

    backend = MultiNoiseEmbeddingBackend()

    stats = {
        "heavy": 0,
        "kappa": 0,
        "lambda": 0,
        "errors": 0,
        "last_recompile_len": (0, 0),
    }

    start_time = time.time()
    pdb_path_base = Path(pdb_dir)

    for idx in tqdm(
        range(start_idx, len(structures)),
        desc="Processing structures",
        initial=start_idx,
        total=len(structures),
    ):
        struct = structures[idx]
        pdb_path = pdb_path_base / f"{struct['id']}.pdb"

        if not pdb_path.exists():
            LOGGER.warning(f"PDB not found: {pdb_path}")
            stats["errors"] += 1
            continue

        try:
            inputs_H = get_inputs(str(pdb_path), chain="H")
            inputs_L = get_inputs(str(pdb_path), chain="L")
        except Exception as e:
            LOGGER.debug(f"Failed to load {pdb_path}: {e}")
            stats["errors"] += 1
            continue

        current_lens = (struct["heavy_len"], struct["light_len"])
        if current_lens != stats["last_recompile_len"]:
            stats["last_recompile_len"] = current_lens

        emb_H = backend.compute_all_noise_levels(
            inputs_H.coords,
            inputs_H.mask,
            inputs_H.chain_ids,
            inputs_H.residue_indices,
            noise_levels,
            idx * 2,
        )
        for noise, emb in emb_H.items():
            update_accumulator(
                accumulators, "heavy", noise, inputs_H.residue_ids, emb
            )
        stats["heavy"] += 1

        emb_L = backend.compute_all_noise_levels(
            inputs_L.coords,
            inputs_L.mask,
            inputs_L.chain_ids,
            inputs_L.residue_indices,
            noise_levels,
            idx * 2 + 1,
        )
        chain_type = "kappa" if struct["locus"] == "K" else "lambda"
        for noise, emb in emb_L.items():
            update_accumulator(
                accumulators, chain_type, noise, inputs_L.residue_ids, emb
            )
        stats[chain_type] += 1

        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, accumulators, idx)
            elapsed = time.time() - start_time
            rate = (idx - start_idx + 1) / elapsed
            remaining = (len(structures) - idx - 1) / rate if rate > 0 else 0
            LOGGER.info(
                f"Progress: {idx + 1}/{len(structures)} "
                f"({rate:.1f} struct/s, ~{remaining / 3600:.1f}h remaining)"
            )

        if (idx + 1) % save_interval == 0:
            save_intermediate_embeddings(
                output_path, accumulators, noise_levels, idx + 1
            )
            LOGGER.info(f"Saved intermediate results at {idx + 1} structures")

    elapsed = time.time() - start_time
    LOGGER.info(f"Processing complete in {elapsed / 3600:.2f} hours")
    LOGGER.info(
        f"Processed: {stats['heavy']} heavy, {stats['kappa']} kappa, "
        f"{stats['lambda']} lambda chains ({stats['errors']} errors)"
    )

    LOGGER.info("Saving final embeddings...")
    save_embeddings_to_dir(
        output_path, accumulators, noise_levels, ["heavy", "kappa", "lambda"]
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        LOGGER.info("Removed checkpoint file after successful completion")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized averaged MPNN embeddings from OAS"
    )
    parser.add_argument(
        "--pdb-dir",
        type=str,
        default=DEFAULT_PDB_DIR,
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to OAS metadata CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of structures to process (for testing)",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_LEVELS,
        help=f"Noise levels in angstroms (default: {DEFAULT_NOISE_LEVELS})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N structures (default: {CHECKPOINT_INTERVAL})",  # noqa: E501
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=SAVE_INTERVAL,
        help=f"Save intermediate results every N (default: {SAVE_INTERVAL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation",
    )
    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("Optimized OAS Structure Embedding Generator")
    LOGGER.info("=" * 60)
    LOGGER.info(f"PDB directory: {args.pdb_dir}")
    LOGGER.info(f"CSV path: {args.csv_path}")
    LOGGER.info(f"Output directory: {args.output_dir}")
    LOGGER.info(f"Noise levels: {args.noise_levels}")
    LOGGER.info(f"Checkpoint interval: {args.checkpoint_interval}")
    LOGGER.info(f"Save interval: {args.save_interval}")
    LOGGER.info(f"Random seed: {args.seed}")
    LOGGER.info(f"JAX devices: {jax.devices()}")
    if args.limit:
        LOGGER.info(f"Processing limit: {args.limit} structures")
    LOGGER.info("=" * 60)

    process_structures(
        args.pdb_dir,
        args.csv_path,
        args.output_dir,
        args.noise_levels,
        limit=args.limit,
        checkpoint_interval=args.checkpoint_interval,
        save_interval=args.save_interval,
    )

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
