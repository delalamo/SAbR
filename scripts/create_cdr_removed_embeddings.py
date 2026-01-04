#!/usr/bin/env python3
"""Create reference embeddings file without CDR/variable positions.

This script creates embeddings_no_cdr.npz by removing variable positions
from the original embeddings.npz. The resulting file contains only
framework positions, used when deterministic_loop_renumbering is enabled.

Variable positions removed:
- CDR1: 27-38
- CDR2: 56-65
- CDR3: 105-117
- Position 10 (missing in lambda/heavy chains)
- DE loop: 81-82 (missing in light chains)

Usage:
    python scripts/create_cdr_removed_embeddings.py
"""

import numpy as np
from pathlib import Path


# All variable positions to remove:
# - CDR1: 27-38, CDR2: 56-65, CDR3: 105-117
# - Position 10 (missing in lambda/heavy chains)
# Note: DE loop positions 81-82 are NOT removed because they are present
# in heavy chains and removing them causes misalignment. FR3 correction
# handles the light chain case where 81-82 are absent.
VARIABLE_POSITIONS = (
    set(range(27, 39)) |    # CDR1: positions 27-38
    set(range(56, 66)) |    # CDR2: positions 56-65
    set(range(105, 118)) |  # CDR3: positions 105-117
    {10}                    # Position 10 (lambda/heavy variability)
    # Note: {81, 82} intentionally NOT removed - handled by FR3 correction
)


def main():
    # Locate the original embeddings file
    script_dir = Path(__file__).parent
    src = script_dir.parent / "src" / "sabr" / "assets" / "embeddings.npz"

    if not src.exists():
        print(f"Error: Original embeddings file not found at {src}")
        return 1

    print(f"Loading original embeddings from: {src}")
    data = np.load(src, allow_pickle=True)

    # Get original data
    idxs = data['idxs']        # IMGT position strings
    array = data['array']      # Embeddings (122, 64)
    stdev = data['stdev']      # Standard deviation (122, 64)
    name = str(data['name'])

    print(f"Original embeddings: {len(idxs)} positions, shape={array.shape}")
    print(f"Original name: {name}")

    # Create mask: True for framework positions, False for variable positions
    keep_mask = np.array([int(idx) not in VARIABLE_POSITIONS for idx in idxs])

    # Filter out variable positions from idxs and array
    new_idxs = idxs[keep_mask]      # Filter out variable position strings
    new_array = array[keep_mask]    # Filter out variable embeddings

    # Set stdev to all ones (uniform scaling, no position-specific normalization)
    new_stdev = np.ones_like(new_array)

    # Report what was removed
    removed_positions = [int(idx) for idx, keep in zip(idxs, keep_mask) if not keep]
    print(f"\nRemoved {len(removed_positions)} variable positions:")
    print(f"  {sorted(removed_positions)}")

    # Report the remaining positions (for verification)
    remaining_positions = [int(idx) for idx in new_idxs]
    print(f"\nRemaining {len(remaining_positions)} framework positions:")
    print(f"  {remaining_positions}")

    # Identify the jumps (where CDRs were removed)
    jumps = []
    for i in range(1, len(remaining_positions)):
        if remaining_positions[i] - remaining_positions[i-1] > 1:
            jumps.append((remaining_positions[i-1], remaining_positions[i]))
    print(f"\nPosition jumps (CDR boundaries): {jumps}")

    # Save new embeddings file
    dst = src.parent / "embeddings_no_cdr.npz"
    np.savez(
        dst,
        array=new_array,
        stdev=new_stdev,
        idxs=new_idxs,
        name='unified_no_cdr'
    )

    print(f"\nCreated: {dst}")
    print(f"  Positions: {len(new_idxs)}")
    print(f"  Array shape: {new_array.shape}")
    print(f"  Stdev shape: {new_stdev.shape} (all ones)")

    # Verify the saved file
    verify = np.load(dst, allow_pickle=True)
    assert len(verify['idxs']) == len(new_idxs), "idxs mismatch"
    assert verify['array'].shape == new_array.shape, "array shape mismatch"
    assert np.allclose(verify['stdev'], 1.0), "stdev should be all ones"
    print("\nVerification: OK")

    return 0


if __name__ == "__main__":
    exit(main())
