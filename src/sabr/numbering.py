"""Convert an IMGT alignment into one of ANARCI's numbering schemes."""

import functools
import logging
from importlib.resources import files

import numpy as np

from sabr import constants
from sabr._anarci.schemes import (
    number_aho,
    number_chothia_heavy,
    number_chothia_light,
    number_imgt,
    number_kabat_heavy,
    number_kabat_light,
    number_martin_heavy,
    number_martin_light,
    number_wolfguy_heavy,
    number_wolfguy_light,
)
from sabr._types import NumberingRecord, NumberingState, NumberingStates

LOGGER = logging.getLogger(__name__)


@functools.cache
def _load_missing_imgt_positions() -> dict[str, frozenset[int]]:
    """Derive absent IMGT positions from the canonical reference metadata.

    The default numbering path uses a full 128-column alignment and never
    needs this metadata. Keeping the asset read behind a cached function makes
    reduced-reference support lazy without introducing a proxy mapping type.
    """
    path = files("sabr.assets") / "embeddings_noise_0.0.npz"
    with (
        path.open("rb") as handle,
        np.load(handle, allow_pickle=True) as archive,
    ):
        data = archive["arr_0"].item()

    all_positions = frozenset(range(1, constants.IMGT_MAX_POSITION + 1))
    return {
        chain_type: all_positions.difference(
            int(position) for position in reference["idxs"]
        )
        for chain_type, reference in data.items()
    }


# These are ANARCI's T-cell receptor chain labels. The public SAbR pipeline
# aligns them against K but retains the actual label for numbering. Of the
# vendored ANARCI schemes, only IMGT and AHo define TCR behavior (and AHo needs
# the actual TCR chain label).
_TCR_CHAIN_TYPES = frozenset(constants.TCR_CHAIN_TYPES)


def _validate_reference_positions(
    matrix: np.ndarray,
    ref_positions: tuple[int, ...],
) -> None:
    """Validate the IMGT-position label for each reference column.

    A normal SAbR alignment has 128 reference columns, so column ``i`` is
    unambiguously IMGT position ``i + 1``. Some low-level callers instead
    align against a reduced reference that omits positions. In that case,
    ``ref_positions[i]`` supplies the absolute IMGT position represented by
    column ``i``.

    There must be exactly one label per matrix column. Labels are immutable
    integer metadata, are confined to one 1--128 IMGT domain, and must be
    strictly increasing so that walking left-to-right through the matrix also
    walks forward through IMGT numbering. ``bool`` is rejected explicitly
    because it is an ``int`` subclass in Python.
    """
    if not isinstance(ref_positions, tuple):
        raise ValueError("ref_positions must be a tuple.")
    if len(ref_positions) != matrix.shape[1]:
        raise ValueError(
            "ref_positions length must match the alignment column count."
        )
    if not all(
        isinstance(position, int) and not isinstance(position, bool)
        for position in ref_positions
    ):
        raise ValueError("ref_positions must contain only integers.")
    if not all(
        1 <= position <= constants.IMGT_MAX_POSITION
        for position in ref_positions
    ):
        raise ValueError("ref_positions must be between IMGT 1 and 128.")
    if any(
        right <= left for left, right in zip(ref_positions, ref_positions[1:])
    ):
        raise ValueError("ref_positions must be strictly increasing.")


def alignment_to_states(
    matrix: np.ndarray,
    *,
    ref_positions: tuple[int, ...] | None = None,
) -> NumberingStates:
    """Translate a query/reference alignment into ANARCI state records.

    ``matrix`` uses query residues as rows and reference positions as columns;
    a value of one assigns that query row to that reference column. Full
    references derive the IMGT position as ``column + 1``. Reduced references
    must provide the absolute position of each column through
    ``ref_positions``.

    ANARCI consumes records shaped like ``((position, state), sequence_index)``
    where ``state`` is ``"m"`` for a match, ``"d"`` for a reference deletion,
    or ``"i"`` for a query insertion. The sequence index is expressed against
    a left-padded sequence: padding makes the first aligned query residue land
    at the zero-based index corresponding to its absolute IMGT position.

    Returns:
        A tuple containing the ordered ANARCI states, the number of leading
        sequence-padding characters required, and the original row of the
        first aligned query residue.

    Raises:
        ValueError: If the matrix is not two-dimensional, contains no aligned
            residues, or has invalid reduced-reference position metadata.
    """
    if matrix.ndim != 2:
        raise ValueError("Alignment matrix must be two-dimensional.")
    if ref_positions is not None:
        _validate_reference_positions(matrix, ref_positions)

    # Transposing makes every path item ``(reference column, query row)``.
    # Sorting in that order lets us emit ANARCI states by IMGT position.
    path = sorted(np.argwhere(matrix.T == 1).tolist())
    if not path:
        raise ValueError("Alignment matrix contains no aligned residues.")

    column_rows = {}
    assigned_rows = set()
    for column, row in path:
        column_rows.setdefault(column, []).append(row)
        assigned_rows.add(row)

    first_column, first_row = path[0]
    last_column, _ = path[-1]
    aligned_columns = sorted(column_rows)
    next_row_by_column = {
        column: column_rows[next_column][0]
        for column, next_column in zip(aligned_columns, aligned_columns[1:])
    }
    first_position = (
        ref_positions[first_column]
        if ref_positions is not None
        else first_column + 1
    )
    imgt_start = first_position - 1
    offset = imgt_start - first_row
    states = []

    for column in range(first_column, last_column + 1):
        imgt_position = (
            ref_positions[column] if ref_positions is not None else column + 1
        )
        if column not in column_rows:
            # No query row occupies this represented reference position.
            states.append(((imgt_position, "d"), None))
            continue

        rows = column_rows[column]
        # The first assigned row matches the reference position. Any further
        # rows assigned to the same column are insertions after that position.
        states.append(((imgt_position, "m"), rows[0] + offset))
        for row in rows[1:]:
            states.append(((imgt_position, "i"), row + offset))

        # Query rows with no matrix assignment, but bracketed by two assigned
        # columns, are also insertions after the earlier IMGT position.
        next_row = next_row_by_column.get(column)
        if next_row is not None:
            for row in range(rows[-1] + 1, next_row):
                if row not in assigned_rows:
                    states.append(((imgt_position, "i"), row + offset))

    return states, imgt_start, first_row


def _insert_missing_deletions(
    states: list[NumberingState], ref_type: str
) -> list[NumberingState]:
    """Add absent reference positions as idempotent deletion states."""
    missing_by_type = _load_missing_imgt_positions()
    if ref_type not in missing_by_type:
        choices = ", ".join(missing_by_type)
        raise ValueError(f"ref_type must be one of {choices}.")

    represented = {state[0][0] for state in states}
    missing = missing_by_type[ref_type].difference(represented)
    if not missing:
        return states

    completed = [
        *states,
        *[((position, "d"), None) for position in sorted(missing)],
    ]
    completed.sort(key=lambda state: state[0][0])
    LOGGER.debug(
        "Inserted %d deletion states for the %s reference.",
        len(missing),
        ref_type,
    )
    return completed


def _apply_scheme(
    states: list[NumberingState], sequence: str, scheme: str, chain_type: str
) -> tuple[list[tuple[tuple[int, str], str]], int, int]:
    """Dispatch to a vendored ANARCI numberer through one call signature."""
    if chain_type in _TCR_CHAIN_TYPES and scheme not in ("imgt", "aho"):
        raise ValueError("TCR chain types support only IMGT or AHo numbering.")

    heavy = chain_type == "H"
    functions = {
        "imgt": number_imgt,
        "aho": lambda current_states, current_sequence: number_aho(
            current_states, current_sequence, chain_type
        ),
        "chothia": (number_chothia_heavy if heavy else number_chothia_light),
        "kabat": (number_kabat_heavy if heavy else number_kabat_light),
        "martin": (number_martin_heavy if heavy else number_martin_light),
        "wolfguy": (number_wolfguy_heavy if heavy else number_wolfguy_light),
    }
    return functions[scheme](states, sequence)


def _number_domain_alignment(
    alignment: np.ndarray,
    sequence: str,
    scheme: str,
    chain_type: str,
    *,
    ref_type: str | None = None,
    ref_positions: tuple[int, ...] | None = None,
) -> list[NumberingRecord]:
    """Return query-row records, optionally restoring a reduced reference."""
    states, imgt_start, first_row = alignment_to_states(
        alignment,
        ref_positions=ref_positions,
    )
    if ref_type is not None:
        missing_by_type = _load_missing_imgt_positions()
        if ref_type not in missing_by_type:
            choices = ", ".join(missing_by_type)
            raise ValueError(f"ref_type must be one of {choices}.")
        if ref_positions is not None:
            missing_positions = frozenset(
                range(1, constants.IMGT_MAX_POSITION + 1)
            ).difference(ref_positions)
            reference_missing = missing_by_type[ref_type]
            if missing_positions != reference_missing:
                raise ValueError(
                    f"ref_positions do not match the {ref_type} reference."
                )
        states = _insert_missing_deletions(states, ref_type)
    padded_sequence = "-" * imgt_start + sequence[first_row:]
    numbered, start, end = _apply_scheme(
        states, padded_sequence, scheme, chain_type
    )
    numbered = [entry for entry in numbered if entry[1] != "-"]
    if not 0 <= start <= end < len(padded_sequence):
        raise ValueError(
            f"ANARCI returned invalid sequence bounds {start}:{end}."
        )

    padded_rows = [None] * imgt_start + list(range(first_row, len(sequence)))
    query_rows = [
        row for row in padded_rows[start : end + 1] if row is not None
    ]
    if len(query_rows) != len(numbered):
        raise ValueError(
            "ANARCI numbering length does not match its query-row span."
        )

    records = []
    for query_row, ((number, insertion_code), amino_acid) in zip(
        query_rows, numbered
    ):
        if sequence[query_row] != amino_acid:
            raise ValueError(
                f"ANARCI expected {amino_acid} at query row {query_row}, "
                f"but the structure contains {sequence[query_row]}."
            )
        records.append((query_row, number, insertion_code, amino_acid))
    LOGGER.info(
        "Numbered %d residues as %s using %s.",
        len(records),
        chain_type,
        scheme,
    )
    return records


def number_alignment(
    alignment: np.ndarray,
    sequence: str,
    scheme: str,
    chain_type: str,
    *,
    ref_type: str | None = None,
    ref_positions: tuple[int, ...] | None = None,
) -> list[NumberingRecord]:
    """Return query-row-to-number records for one alignment.

    ``ref_type`` and ``ref_positions`` are opt-in metadata for reduced
    single-domain references. Normal full-width single- and multi-domain
    alignments leave them unset.

    Rows index the Q characters of ``sequence``; columns are absolute IMGT
    positions in a [Q, 128 * D] matrix for D domains, or [Q, R] for an explicit
    reduced reference. Returns a list of (query row, number, insertion code,
    amino acid) tuples. Query rows are zero-based; assigned residue numbers
    include the numbering stride for domains after the first.
    """
    chain_types = tuple(chain_type)
    is_multidomain = len(chain_types) > 1
    if is_multidomain and (ref_type is not None or ref_positions is not None):
        raise ValueError(
            "Reference metadata is supported only for single-domain "
            "alignments."
        )

    if is_multidomain:
        expected_columns = len(chain_types) * constants.IMGT_MAX_POSITION
        if alignment.ndim != 2 or alignment.shape[1] != expected_columns:
            raise ValueError(
                f"Multi-domain alignment must have {expected_columns} "
                "columns."
            )

    domains = []
    for domain_index, domain_type in enumerate(chain_types):
        if is_multidomain:
            start = domain_index * constants.IMGT_MAX_POSITION
            end = start + constants.IMGT_MAX_POSITION
            domain_alignment = alignment[:, start:end]
        else:
            domain_alignment = alignment
        records = _number_domain_alignment(
            domain_alignment,
            sequence,
            scheme,
            domain_type,
            ref_type=ref_type,
            ref_positions=ref_positions,
        )
        if domain_index:
            number_offset = domain_index * constants.DOMAIN_NUMBERING_STRIDE
            records = [
                (
                    query_row,
                    number + number_offset,
                    insertion_code,
                    amino_acid,
                )
                for query_row, number, insertion_code, amino_acid in records
            ]
        domains.append(records)

    if not is_multidomain:
        return domains[0]

    records = list(domains[0])
    for previous_domain, next_domain in zip(domains, domains[1:]):
        linker_start = previous_domain[-1][0] + 1
        linker_end = next_domain[0][0]
        records.extend(
            (
                query_row,
                previous_domain[-1][1] + linker_index,
                "",
                sequence[query_row],
            )
            for linker_index, query_row in enumerate(
                range(linker_start, linker_end), start=1
            )
        )
        records.extend(next_domain)
    LOGGER.info(
        "Numbered %d residues as multi-domain %s using %s.",
        len(records),
        chain_type,
        scheme,
    )
    return records
