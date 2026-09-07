"""Reference alignment using the validated affine Smith-Waterman method."""

import functools
import logging
from importlib.resources import files

import numpy as np

from sabr import constants
from sabr.corrections import apply_corrections

LOGGER = logging.getLogger(__name__)


def _soft_maximum_with_grad(
    x, temperature, ninf, axis=-1, mask=None, *, with_gradient=True
):
    """Log-sum-exp and optional derivative, including historical clipping."""
    scaled = x / temperature
    clipped = np.maximum(scaled, ninf)
    maximum = np.max(clipped, axis=axis, keepdims=True)
    weights = np.exp(clipped - maximum)
    if mask is not None:
        weights *= mask
    total = np.sum(weights, axis=axis, keepdims=True)
    value = temperature * (maximum + np.log(total))
    if with_gradient:
        weights /= total
        weights *= np.where(scaled > ninf, 1, np.where(scaled == ninf, 0.5, 0))
    return np.squeeze(value, axis=axis), weights if with_gradient else None


def _affine_value_and_grad(
    similarities,
    lengths,
    temperature,
    gap_extend,
    gap_open,
    free_gap_boundary=-1,
    NINF=-1e30,
    *,
    with_gradient=True,
):
    """Striped affine DP with an explicit reverse pass for soft assignments.

    This differentiates the same three-state recurrence as the JAX version;
    it does not replace soft alignment with a hard traceback. Forward and
    reverse sweeps vectorize across each antidiagonal.
    Score-only calls omit transition storage and return None for the gradient.
    """
    similarities = np.asarray(similarities, dtype=np.float32)
    rows, columns = similarities.shape
    mask = (np.arange(rows) < lengths[0])[:, None] & (
        np.arange(columns) < lengths[1]
    )[None, :]
    x = similarities + np.float32(NINF) * (1 - mask).astype(np.float32)
    a, b = rows - 1, columns - 1
    ar, br = np.arange(a)[::-1, None], np.arange(b)[None, :]
    ii, jj = br - ar + a - 1, (ar + br) // 2
    count, width = a + b - 1, (a + b) // 2
    stripes = np.full((count, width), NINF, dtype=np.float32)
    stripes[ii, jj] = x[:-1, :-1]
    free = np.zeros((count, width), dtype=bool)
    free[ii, jj] = np.broadcast_to(
        np.isin(br, np.atleast_1d(free_gap_boundary)), (a, b)
    )
    states = np.full((count + 2, width, 3), NINF, dtype=np.float32)
    # Match, right and down states have four, two and three predecessors.
    transitions = (
        tuple(
            np.empty((count, width, size), dtype=np.float32)
            for size in (4, 2, 3)
        )
        if with_gradient
        else ()
    )
    right_penalty = np.asarray((gap_open, gap_extend, gap_open), np.float32)
    down_penalty = np.asarray((gap_open, gap_open, gap_extend), np.float32)
    aligned = np.empty((width, 4), dtype=np.float32)
    right = np.empty((width, 3), dtype=np.float32)
    down = np.empty((width, 3), dtype=np.float32)
    for index in range(count):
        h2, h1 = states[index], states[index + 1]
        aligned[:, :3] = h2
        aligned[:, 3] = 0
        aligned += stripes[index, :, None]
        odd = (index + a % 2) % 2
        if odd:
            right[0] = NINF
            right[1:] = h1[:-1]
            down[:] = h1
        else:
            right[:] = h1
            down[:-1] = h1[1:]
            down[-1] = NINF
        right += right_penalty
        down += np.where(free[index, :, None], 0, down_penalty)
        for state, candidates in enumerate((aligned, right[:, :2], down)):
            value, weights = _soft_maximum_with_grad(
                candidates, temperature, NINF, with_gradient=with_gradient
            )
            states[index + 2, :, state] = value
            if with_gradient:
                transitions[state][index] = weights
    endings = states[ii + 2, jj] + x[1:, 1:, None]
    score, end_grad = _soft_maximum_with_grad(
        endings,
        temperature,
        NINF,
        axis=None,
        mask=mask[1:, 1:, None],
        with_gradient=with_gradient,
    )
    if not with_gradient:
        return score, None
    adjoints = np.zeros_like(states)
    adjoints[ii + 2, jj] = end_grad
    gradient = np.zeros_like(x)
    gradient[1:, 1:] = np.sum(end_grad, axis=-1)
    stripe_gradient = np.zeros_like(stripes)
    for index in range(count - 1, -1, -1):
        incoming = adjoints[index + 2]
        aligned, right, down = (
            transition[index] * incoming[:, state, None]
            for state, transition in enumerate(transitions)
        )
        stripe_gradient[index] = np.sum(aligned, axis=-1)
        adjoints[index] += aligned[:, :3]
        if (index + a % 2) % 2:
            adjoints[index + 1, :-1, :2] += right[1:]
            adjoints[index + 1] += down
        else:
            adjoints[index + 1, :, :2] += right
            adjoints[index + 1, 1:] += down[:-1]
    gradient[:-1, :-1] += stripe_gradient[ii, jj]
    return score, gradient


def _affine_score(
    similarities,
    lengths,
    temperature,
    gap_extend,
    gap_open,
    free_gap_boundary=-1,
    NINF=-1e30,
):
    """Return the historical affine soft-alignment score."""
    return _affine_value_and_grad(
        similarities,
        lengths,
        temperature,
        gap_extend,
        gap_open,
        free_gap_boundary,
        NINF,
        with_gradient=False,
    )[0]


@functools.cache
def load_references(
    noise_level: float,
    mode: str = "sabr",
    scfv: bool = False,
) -> dict:
    """Load single-domain and optional scFv references for one mode."""
    if mode == "sabr":
        filename = f"embeddings_noise_{noise_level:.1f}.npz"
    elif mode == "softalign":
        filename = "softalign_embeddings.npz"
    else:
        raise ValueError(f"mode must be one of {constants.MODES}.")
    path = files("sabr.assets") / filename
    with path.open("rb") as handle:
        data = np.load(handle, allow_pickle=True)["arr_0"].item()
    references = {}
    for chain_type, reference in data.items():
        embeddings = np.asarray(reference["array"])
        embeddings.flags.writeable = False
        references[chain_type] = (
            embeddings,
            tuple(int(position) for position in reference["idxs"]),
        )

    if scfv:
        for first_type, second_type in constants.SCFV_CHAIN_TYPES:
            first_embeddings, first_positions = references[first_type]
            second_embeddings, second_positions = references[second_type]
            embeddings = np.concatenate(
                (first_embeddings, second_embeddings), axis=0
            )
            embeddings.flags.writeable = False
            references[first_type + second_type] = (
                embeddings,
                (
                    *first_positions,
                    *(
                        position + constants.IMGT_MAX_POSITION
                        for position in second_positions
                    ),
                ),
            )
    return references


@functools.cache
def load_gap_penalties(mode: str = "sabr") -> tuple[float, float]:
    """Return the gap extension and opening values for one mode."""
    if mode == "sabr":
        return constants.SW_GAP_EXTEND, constants.SW_GAP_OPEN
    if mode != "softalign":
        raise ValueError(f"mode must be one of {constants.MODES}.")
    path = files("sabr.assets") / "softalign_gap.npz"
    with path.open("rb") as handle:
        data = np.load(handle, allow_pickle=False)
        return float(data["gap_extend"]), float(data["gap_open"])


def _alignment_path(alignment: np.ndarray) -> np.ndarray:
    """Validate common matrix invariants and return its assigned path."""
    if alignment.ndim != 2 or not np.isfinite(alignment).all():
        raise ValueError("Alignment must be a finite two-dimensional matrix.")
    if not np.isin(alignment, (0, 1)).all():
        raise ValueError("Alignment matrix must contain only zeroes and ones.")
    if np.any(alignment.sum(axis=1) > 1):
        raise ValueError("Alignment assigns a query row more than once.")
    if np.any(alignment.sum(axis=0) > 1):
        raise ValueError("Alignment assigns an IMGT column more than once.")

    path = np.argwhere(alignment == 1)
    if not len(path):
        raise ValueError("Alignment contains no assigned residues.")
    rows = path[:, 0]
    columns = path[:, 1]
    if np.any(np.diff(rows) <= 0) or np.any(np.diff(columns) <= 0):
        raise ValueError(
            "Alignment assignments are not strictly monotonic; use "
            "residue_range to select one antibody domain."
        )
    return path


def _validate_multidomain_alignment(
    alignment: np.ndarray, representation: str
) -> None:
    """Validate ordered IMGT-domain blocks with optional linkers."""
    expected_columns = len(representation) * constants.IMGT_MAX_POSITION
    if alignment.ndim != 2 or alignment.shape[1] != expected_columns:
        raise ValueError(
            f"Multi-domain alignment must have {expected_columns} columns."
        )
    _alignment_path(alignment)

    for domain_index, chain_type in enumerate(representation):
        start = domain_index * constants.IMGT_MAX_POSITION
        end = start + constants.IMGT_MAX_POSITION
        domain = alignment[:, start:end]
        assigned_rows = np.flatnonzero(domain.sum(axis=1))
        if not len(assigned_rows):
            raise ValueError(
                f"Multi-domain {chain_type} domain {domain_index + 1} "
                "contains no assigned residues."
            )


def _concatenate_reference(references: dict, representation: str) -> tuple:
    """Build one ordered multi-domain reference from single domains."""
    domain_references = [references[domain] for domain in representation]
    embeddings = np.concatenate(
        [reference[0] for reference in domain_references], axis=0
    )
    embeddings.flags.writeable = False
    positions = tuple(
        position + domain_index * constants.IMGT_MAX_POSITION
        for domain_index, (_, domain_positions) in enumerate(domain_references)
        for position in domain_positions
    )
    return embeddings, positions


def _align_reference(
    query: np.ndarray,
    reference: np.ndarray,
    mode: str = "sabr",
    free_gap_boundary: int | tuple[int, ...] = -1,
    *,
    score_only: bool = False,
):
    """Align one reference, or return only its similarity matrix and score."""
    anchor = np.zeros((1, reference.shape[1]), dtype=reference.dtype)
    augmented_reference = np.concatenate((anchor, reference, anchor), axis=0)
    query = np.asarray(query, dtype=np.float32)
    augmented_reference = np.asarray(augmented_reference, dtype=np.float32)
    similarity = np.einsum("ia,ja->ij", query, augmented_reference)
    lengths = np.asarray(similarity.shape)
    gap_extend, gap_open = load_gap_penalties(mode)
    score, soft_alignment = _affine_value_and_grad(
        similarity,
        lengths,
        constants.DEFAULT_TEMPERATURE,
        gap_extend,
        gap_open,
        free_gap_boundary,
        with_gradient=not score_only,
    )
    return (
        None if score_only else soft_alignment[:, 1:-1],
        similarity,
        float(score),
    )


def _affine_gap_penalty(
    length: int,
    gap_extend: float = constants.SW_GAP_EXTEND,
    gap_open: float = constants.SW_GAP_OPEN,
) -> float:
    """Return the normal affine score for one gap of the given length."""
    if length <= 0:
        return 0.0
    return gap_open + (length - 1) * gap_extend


def _terminal_gap_penalty(
    alignment: np.ndarray,
    mode: str = "sabr",
) -> float:
    """Score unaligned query and reference termini as affine gaps."""
    discrete = np.round(alignment).astype(int)
    path = np.argwhere(discrete == 1)
    if not len(path):
        raise ValueError("Alignment contains no assigned residues.")

    first_query, first_reference = path[0]
    last_query, last_reference = path[-1]
    terminal_lengths = (
        int(first_query),
        int(first_reference),
        alignment.shape[0] - int(last_query) - 1,
        alignment.shape[1] - int(last_reference) - 1,
    )
    gap_extend, gap_open = load_gap_penalties(mode)
    return sum(
        _affine_gap_penalty(length, gap_extend, gap_open)
        for length in terminal_lengths
    )


def align(
    query: np.ndarray,
    gap_indices: frozenset,
    chain_type: str,
    noise_level: float,
    mode: str = "sabr",
    scfv: bool = False,
) -> tuple:
    """Align query embeddings and return corrected IMGT alignment metadata."""
    references = dict(load_references(noise_level, mode, scfv=scfv))
    if chain_type == "auto":
        candidates = (
            constants.SCFV_CHAIN_TYPES if scfv else constants.CHAIN_TYPES
        )
    elif chain_type in constants.TCR_CHAIN_TYPES:
        candidates = ("K",)
    else:
        candidates = tuple(chain_type.split(","))
    # Single-domain selection depends only on scores. Multi-domain candidates
    # need their alignments to calculate terminal gap penalties.
    # Three candidates amortize recomputing the winner's forward pass.
    score_only = len(candidates) > 2 and all(
        len(candidate) == 1 for candidate in candidates
    )
    best = None
    for candidate in candidates:
        if candidate not in references:
            references[candidate] = _concatenate_reference(
                references, candidate
            )
        reference, positions = references[candidate]
        if len(candidate) > 1:
            free_gap_boundaries = tuple(
                sum(
                    position <= domain_index * constants.IMGT_MAX_POSITION
                    for position in positions
                )
                for domain_index in range(1, len(candidate))
            )
            reduced, similarity, score = _align_reference(
                query,
                reference,
                mode,
                free_gap_boundary=free_gap_boundaries,
            )
        elif score_only:
            reduced, similarity, score = _align_reference(
                query, reference, mode, score_only=True
            )
        else:
            reduced, similarity, score = _align_reference(
                query, reference, mode
            )
        if (
            not np.isfinite(score)
            or (reduced is not None and not np.isfinite(reduced).all())
            or not np.isfinite(similarity).all()
        ):
            raise ValueError(
                f"{candidate} reference produced a non-finite alignment."
            )
        selection_score = score
        if len(candidate) > 1:
            terminal_penalty = _terminal_gap_penalty(reduced, mode)
            selection_score += terminal_penalty
            LOGGER.info(
                "%s reference score: %.4f; terminal gap penalty: %.4f; "
                "selection score: %.4f",
                candidate,
                score,
                terminal_penalty,
                selection_score,
            )
        else:
            LOGGER.info("%s reference score: %.4f", candidate, score)
        if best is None or selection_score > best[0]:
            best = (
                selection_score,
                score,
                candidate,
                reduced,
                positions,
            )

    _, score, selected_type, reduced, positions = best
    if score_only:
        reduced, _, _ = _align_reference(
            query, references[selected_type][0], mode
        )
        if not np.isfinite(reduced).all():
            raise ValueError(
                f"{selected_type} reference produced a non-finite alignment."
            )
    domain_count = len(selected_type)
    full_alignment = np.zeros(
        (query.shape[0], domain_count * constants.IMGT_MAX_POSITION),
        dtype=reduced.dtype,
    )
    full_alignment[:, np.asarray(positions) - 1] = reduced
    full_alignment = np.round(full_alignment).astype(int)
    if len(selected_type) > 1:
        corrected = full_alignment.copy()
        for domain_index, _ in enumerate(selected_type):
            start = domain_index * constants.IMGT_MAX_POSITION
            end = start + constants.IMGT_MAX_POSITION
            corrected[:, start:end] = apply_corrections(
                corrected[:, start:end],
                gap_indices=gap_indices,
            )
        _validate_multidomain_alignment(corrected, selected_type)
    else:
        corrected = apply_corrections(full_alignment, gap_indices=gap_indices)
        _alignment_path(corrected)
    return corrected, selected_type, score
