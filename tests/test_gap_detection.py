#!/usr/bin/env python3
"""Tests for structural gap detection functionality."""

import numpy as np
import pytest

from sabr import constants
from sabr.util import detect_backbone_gaps, has_gap_in_region


class TestDetectBackboneGaps:
    """Tests for detect_backbone_gaps function."""

    def test_uses_correct_atom_indices(self):
        """Verify gap detection uses C (index 2) and N (index 0), not CA or CB.

        Backbone atom order is [N, CA, C, CB] with indices [0, 1, 2, 3].
        Gap detection should measure distance from C of residue i to N of i+1.
        """
        # Create 2 residues where only C->N distance indicates a gap
        coords = np.zeros((2, 4, 3))

        # Residue 0: Place atoms at different positions
        coords[0, 0, :] = [0, 0, 0]  # N atom
        coords[0, 1, :] = [1, 0, 0]  # CA atom
        coords[0, 2, :] = [2, 0, 0]  # C atom - THIS should be used
        coords[0, 3, :] = [3, 0, 0]  # CB atom

        # Residue 1: N atom far from residue 0's C atom (gap)
        coords[1, 0, :] = [5, 0, 0]  # N atom - THIS should be used (dist=3)
        coords[1, 1, :] = [6, 0, 0]  # CA atom
        coords[1, 2, :] = [7, 0, 0]  # C atom
        coords[1, 3, :] = [8, 0, 0]  # CB atom

        # C[0] to N[1] distance = |5 - 2| = 3 Å (gap)
        gaps = detect_backbone_gaps(coords, threshold=2.5)

        assert 0 in gaps, "Should detect gap using C->N distance (3 Å > 2.5 Å)"

    def test_no_gap_when_atoms_close(self):
        """Verify no gap detected when C-N distance is within threshold."""
        coords = np.zeros((2, 4, 3))

        # Normal peptide bond: C-N distance ~1.33 Å
        coords[0, 2, :] = [0, 0, 0]  # C atom of residue 0
        coords[1, 0, :] = [1.33, 0, 0]  # N atom of residue 1

        gaps = detect_backbone_gaps(coords)

        assert len(gaps) == 0, "No gap should be detected for normal bond"

    def test_gap_detected_above_threshold(self):
        """Verify gap is detected when C-N distance exceeds threshold."""
        coords = np.zeros((2, 4, 3))

        # Gap: C-N distance = 5 Å (well above 2.0 Å threshold)
        coords[0, 2, :] = [0, 0, 0]  # C atom of residue 0
        coords[1, 0, :] = [5, 0, 0]  # N atom of residue 1

        gaps = detect_backbone_gaps(coords)

        assert 0 in gaps, "Gap should be detected for 5 Å distance"

    def test_gap_at_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        coords = np.zeros((2, 4, 3))

        # Exactly at threshold (2.0 Å) - should NOT be a gap
        coords[0, 2, :] = [0, 0, 0]
        coords[1, 0, :] = [2.0, 0, 0]

        gaps = detect_backbone_gaps(coords, threshold=2.0)
        assert 0 not in gaps, "Exact threshold should not be a gap"

        # Slightly above threshold - should be a gap
        coords[1, 0, :] = [2.01, 0, 0]
        gaps = detect_backbone_gaps(coords, threshold=2.0)
        assert 0 in gaps, "Above threshold should be a gap"

    def test_multiple_gaps(self):
        """Verify detection of multiple gaps in a chain."""
        coords = np.zeros((5, 4, 3))

        # Residue 0: C at origin
        coords[0, 2, :] = [0, 0, 0]
        # Residue 1: N close (no gap), C for next check
        coords[1, 0, :] = [1.3, 0, 0]
        coords[1, 2, :] = [2.6, 0, 0]
        # Residue 2: N far (gap at index 1)
        coords[2, 0, :] = [10, 0, 0]
        coords[2, 2, :] = [11.3, 0, 0]
        # Residue 3: N close (no gap)
        coords[3, 0, :] = [12.6, 0, 0]
        coords[3, 2, :] = [13.9, 0, 0]
        # Residue 4: N far (gap at index 3)
        coords[4, 0, :] = [20, 0, 0]

        gaps = detect_backbone_gaps(coords)

        assert 1 in gaps, "Gap should be detected between residues 1 and 2"
        assert 3 in gaps, "Gap should be detected between residues 3 and 4"
        assert 0 not in gaps, "No gap between residues 0 and 1"
        assert 2 not in gaps, "No gap between residues 2 and 3"

    def test_custom_threshold(self):
        """Verify custom threshold is respected."""
        coords = np.zeros((2, 4, 3))

        # Distance of 3 Å
        coords[0, 2, :] = [0, 0, 0]
        coords[1, 0, :] = [3, 0, 0]

        # With default threshold (2.0), should be a gap
        gaps_default = detect_backbone_gaps(coords)
        assert 0 in gaps_default

        # With higher threshold (4.0), should NOT be a gap
        gaps_high = detect_backbone_gaps(coords, threshold=4.0)
        assert 0 not in gaps_high

    def test_3d_distance_calculation(self):
        """Verify proper 3D Euclidean distance calculation."""
        coords = np.zeros((2, 4, 3))

        # Distance in 3D: sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ≈ 1.73 Å
        coords[0, 2, :] = [0, 0, 0]
        coords[1, 0, :] = [1, 1, 1]

        gaps = detect_backbone_gaps(coords, threshold=1.5)
        assert 0 in gaps, "3D distance ~1.73 should exceed 1.5 threshold"

        gaps = detect_backbone_gaps(coords, threshold=2.0)
        assert (
            0 not in gaps
        ), "3D distance ~1.73 should not exceed 2.0 threshold"

    def test_handles_batch_dimension(self):
        """Verify function handles [1, N, 4, 3] input shape."""
        coords = np.zeros((1, 2, 4, 3))  # With batch dimension

        coords[0, 0, 2, :] = [0, 0, 0]  # C atom
        coords[0, 1, 0, :] = [5, 0, 0]  # N atom (gap)

        gaps = detect_backbone_gaps(coords)

        assert 0 in gaps, "Should handle batch dimension input"

    def test_invalid_shape_raises_error(self):
        """Verify proper error for invalid input shapes."""
        with pytest.raises(ValueError, match="Expected coords shape"):
            detect_backbone_gaps(np.zeros((2, 3, 3)))  # Wrong atom count

        with pytest.raises(ValueError, match="Expected coords shape"):
            detect_backbone_gaps(np.zeros((2, 4, 2)))  # Wrong xyz count

    def test_single_residue_returns_empty(self):
        """Verify single residue returns empty set (no gaps possible)."""
        coords = np.zeros((1, 4, 3))

        gaps = detect_backbone_gaps(coords)

        assert len(gaps) == 0, "Single residue cannot have gaps"

    def test_default_threshold_matches_constant(self):
        """Verify default threshold uses PEPTIDE_BOND_MAX_DISTANCE constant."""
        assert constants.PEPTIDE_BOND_MAX_DISTANCE == 2.0

        coords = np.zeros((2, 4, 3))
        coords[0, 2, :] = [0, 0, 0]
        coords[1, 0, :] = [2.5, 0, 0]  # Above 2.0 Å

        # Using default should detect gap
        gaps = detect_backbone_gaps(coords)
        assert 0 in gaps

    def test_returns_frozenset(self):
        """Verify function returns a FrozenSet for immutability."""
        coords = np.zeros((2, 4, 3))
        coords[0, 2, :] = [0, 0, 0]
        coords[1, 0, :] = [5, 0, 0]

        gaps = detect_backbone_gaps(coords)

        assert isinstance(gaps, frozenset), "Should return a FrozenSet"

    def test_vectorized_performance(self):
        """Verify function handles large inputs efficiently."""
        # Create a chain with 1000 residues
        n_residues = 1000
        coords = np.zeros((n_residues, 4, 3))

        # Set up normal peptide bonds (C-N distance ~1.3 Å)
        for i in range(n_residues):
            coords[i, 0, :] = [i * 3.8, 0, 0]  # N
            coords[i, 2, :] = [i * 3.8 + 2.5, 0, 0]  # C

        # Add a gap at position 500
        coords[501, 0, :] = [501 * 3.8 + 10, 0, 0]

        gaps = detect_backbone_gaps(coords)

        assert 500 in gaps, "Should detect gap at position 500"
        assert len(gaps) == 1, "Should only detect one gap"


class TestHasGapInRegion:
    """Tests for has_gap_in_region function."""

    def test_gap_within_region(self):
        """Verify detection when gap is within region bounds."""
        gap_indices = frozenset({5})

        assert has_gap_in_region(gap_indices, 3, 8) is True
        assert has_gap_in_region(gap_indices, 5, 10) is True

    def test_gap_outside_region(self):
        """Verify no detection when gap is outside region bounds."""
        gap_indices = frozenset({5})

        # Region [0,5]: residues 0-5, check gaps 0-4. Gap 5 is between 5-6.
        assert has_gap_in_region(gap_indices, 0, 5) is False
        assert has_gap_in_region(gap_indices, 6, 10) is False
        assert has_gap_in_region(gap_indices, 10, 15) is False

    def test_multiple_gaps_one_in_region(self):
        """Verify detection when one of multiple gaps is in region."""
        gap_indices = frozenset({2, 8, 15})

        # Region [5,10]: check gaps 5-9. Gap 8 is in range.
        assert has_gap_in_region(gap_indices, 5, 10) is True
        # Region [0,3]: check gaps 0-2. Gap 2 is in range.
        assert has_gap_in_region(gap_indices, 0, 3) is True
        # Region [3,8]: check gaps 3-7. Neither 2 nor 8 is in range.
        assert has_gap_in_region(gap_indices, 3, 8) is False

    def test_empty_gap_set(self):
        """Verify no detection with empty gap set."""
        gap_indices = frozenset()

        assert has_gap_in_region(gap_indices, 0, 100) is False

    def test_region_boundary_behavior(self):
        """Verify gap at end_row is NOT checked (it's outside the region)."""
        gap_indices = frozenset({5})

        # Gap at index 5 = break between residues 5 and 6.
        # Region [5, 10] (inclusive): residues 5,6,7,8,9,10
        # Internal gaps are indices 5,6,7,8,9 (between consecutive pairs)
        # Gap 5 (between 5-6) is internal, so detected.
        assert has_gap_in_region(gap_indices, 5, 10) is True

        # Region [0, 5] (inclusive): residues 0,1,2,3,4,5
        # Internal gaps are indices 0,1,2,3,4 (between consecutive pairs)
        # Gap 5 (between 5-6) is NOT internal since 6 is outside region.
        assert has_gap_in_region(gap_indices, 0, 5) is False

    def test_single_residue_region(self):
        """Verify behavior with two-residue region."""
        gap_indices = frozenset({5})

        # Region [5, 6] (inclusive): residues 5,6. Gap 5 is between them.
        assert has_gap_in_region(gap_indices, 5, 6) is True
        # Region [4, 5] (inclusive): residues 4,5. Gap 5 is between 5-6.
        assert has_gap_in_region(gap_indices, 4, 5) is False
