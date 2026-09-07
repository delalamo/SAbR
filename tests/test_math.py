import hashlib
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import PDBParser

from sabr import constants
from sabr.alignment import (
    _affine_gap_penalty,
    _affine_score,
    _affine_value_and_grad,
    _align_reference,
    _alignment_path,
    _concatenate_reference,
    _terminal_gap_penalty,
    _validate_multidomain_alignment,
    align,
    load_gap_penalties,
    load_references,
)
from sabr.model import encode, load_parameters
from sabr.numbering import _load_missing_imgt_positions, alignment_to_states
from sabr.structure import extract_chain

DATA = Path(__file__).parent / "data"
ASSETS = Path(__file__).parents[1] / "src" / "sabr" / "assets"


def test_scientific_assets_are_unchanged():
    expected = {
        "mpnn_encoder.npz": (
            "ee70ee43b4c9fdcf7528ad1c7ccfcba20"
            "f47d99de7568939db2f034f21b37ca7"
        ),
        "embeddings_noise_0.0.npz": (
            "1a738d26d0decb75f2fe0b83da1c71ac"
            "91433d80cc3d8270e669e99cf00e24bb"
        ),
        "embeddings_noise_0.2.npz": (
            "619286656b05d0921a4bf7a587bfc767"
            "d93f9e193fd2d2ff6787d7a4b7fa9ef3"
        ),
        "embeddings_noise_0.5.npz": (
            "3a4f51604dd0b2d27352ad269ed9ceb62"
            "42d08dc0a837bf5d3846af145a4444c"
        ),
        "embeddings_noise_1.0.npz": (
            "70495a54ae02a4db70ade0be2252c86b"
            "b50b4b02197b9ac33b042493dad88deb"
        ),
        "embeddings_noise_2.0.npz": (
            "b2f8e886e73655d25c68d23488ad17a"
            "2e441af9fe016da50d28b3e05928861e2"
        ),
        "softalign_encoder.npz": (
            "eb25b5b37b4aa62c0d70cfea78e7fd32"
            "8e7fbfce2b05a783f19b701145edba32"
        ),
        "softalign_embeddings.npz": (
            "640415ee69f144cb6d4a6ab0df2817bd0"
            "541dc8b0252f40c14e312f2e9448533"
        ),
        "softalign_gap.npz": (
            "ba3565786a7d9d3e69f6a41796f3b40"
            "e361ec65ed33cb4b435019549f2b5abae"
        ),
    }
    for filename, digest in expected.items():
        assert (
            hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest()
            == digest
        )
    assert not (ASSETS / "embeddings.npz").exists()


def test_encoder_and_affine_alignment_match_captured_main_baseline():
    baseline = np.load(DATA / "math_baseline.npz")
    structure = PDBParser(QUIET=True).get_structure(
        "heavy", DATA / "test_heavy_chain.pdb"
    )
    data = extract_chain(structure, "F", None)
    embeddings = encode(data.coords)
    np.testing.assert_allclose(
        embeddings, baseline["embeddings"], rtol=1e-5, atol=3e-6
    )
    softalign_embeddings = encode(data.coords, "softalign")
    assert softalign_embeddings.shape == embeddings.shape
    assert not np.allclose(softalign_embeddings, embeddings)

    reference, positions = load_references(0.0)["H"]
    reduced, similarity, score = _align_reference(embeddings, reference)
    np.testing.assert_allclose(
        reduced, baseline["reduced_alignment"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        similarity, baseline["similarity"], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(score, baseline["score"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(
        np.round(reduced), np.round(baseline["reduced_alignment"])
    )
    np.testing.assert_array_equal(positions, baseline["positions"])


@pytest.mark.parametrize("shape", ((4, 5), (5, 4)))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("boundary", (-1, 1, (1, 2)))
def test_affine_reverse_pass_matches_finite_differences(
    shape, padded, boundary
):
    similarities = (
        np.random.default_rng(214).normal(size=shape).astype(np.float32)
    )
    lengths = np.asarray(shape) - int(padded)
    arguments = (lengths, 0.7, -0.5, -2.0, boundary)
    _, gradient = _affine_value_and_grad(similarities, *arguments)
    numerical = np.empty_like(similarities)
    step = 0.002
    for index in np.ndindex(shape):
        plus = similarities.copy()
        minus = similarities.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            _affine_score(plus, *arguments) - _affine_score(minus, *arguments)
        ) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, rtol=0.003, atol=0.0003)


def test_every_noise_asset_has_all_chain_references():
    for level in constants.NOISE_LEVELS:
        references = load_references(level)
        assert tuple(references) == constants.CHAIN_TYPES
        for embeddings, positions in references.values():
            assert embeddings.shape[1] == constants.EMBED_DIM
            assert embeddings.shape[0] == len(positions)


def test_missing_imgt_metadata_matches_every_reference_asset():
    reference_sets = [
        load_references(level) for level in constants.NOISE_LEVELS
    ]
    reference_sets.append(load_references(0.0, "softalign"))
    missing_imgt_positions = _load_missing_imgt_positions()

    all_positions = frozenset(range(1, constants.IMGT_MAX_POSITION + 1))
    for references in reference_sets:
        for chain_type, (_, positions) in references.items():
            assert all_positions.difference(positions) == (
                missing_imgt_positions[chain_type]
            )


@pytest.mark.parametrize(
    ("mode", "noise_level"),
    (("sabr", 0.0), ("softalign", 2.0)),
)
def test_scfv_references_are_exact_ordered_concatenations(mode, noise_level):
    references = load_references(noise_level, mode, scfv=True)
    assert tuple(references) == (
        *constants.CHAIN_TYPES,
        *constants.SCFV_CHAIN_TYPES,
    )
    for representation in constants.SCFV_CHAIN_TYPES:
        first_type, second_type = representation
        embeddings, positions = references[representation]
        first_embeddings, first_positions = references[first_type]
        second_embeddings, second_positions = references[second_type]
        np.testing.assert_array_equal(
            embeddings,
            np.concatenate((first_embeddings, second_embeddings), axis=0),
        )
        assert positions == (
            *first_positions,
            *(
                position + constants.IMGT_MAX_POSITION
                for position in second_positions
            ),
        )
        assert not embeddings.flags.writeable


def test_scfv_alignment_allows_an_unassigned_linker_between_domains():
    alignment = np.zeros((5, 256), dtype=int)
    alignment[0, 0] = 1
    alignment[1, 1] = 1
    alignment[3, 128] = 1
    alignment[4, 129] = 1
    _validate_multidomain_alignment(alignment, "HK")


def test_higher_order_reference_is_an_ordered_concatenation():
    references = load_references(0.0)
    embeddings, positions = _concatenate_reference(references, "HHK")
    expected_embeddings = np.concatenate(
        (
            references["H"][0],
            references["H"][0],
            references["K"][0],
        ),
        axis=0,
    )
    np.testing.assert_array_equal(embeddings, expected_embeddings)
    assert positions == tuple(
        position + domain_index * constants.IMGT_MAX_POSITION
        for domain_index, domain_type in enumerate("HHK")
        for position in references[domain_type][1]
    )
    assert not embeddings.flags.writeable


@pytest.mark.parametrize("linker_length", (1, 3))
def test_scfv_reference_boundary_has_no_affine_linker_penalty(linker_length):
    query_length = linker_length + 2
    similarities = np.full((query_length, 4), -100.0, dtype=np.float32)
    similarities[:, (0, 3)] = 0.0
    similarities[0, 1] = 10.0
    similarities[-1, 2] = 10.0
    lengths = np.asarray((query_length, similarities.shape[1]))
    gap_extend = -1.0
    gap_open = -3.0

    penalized = _affine_score(
        similarities,
        lengths,
        1e-4,
        gap_extend,
        gap_open,
    )
    free = _affine_score(
        similarities,
        lengths,
        1e-4,
        gap_extend,
        gap_open,
        free_gap_boundary=1,
    )

    expected_penalty = gap_open + (linker_length - 1) * gap_extend
    assert float(penalized) == pytest.approx(20.0 + expected_penalty)
    assert float(free) == pytest.approx(20.0)


def test_higher_order_reference_makes_every_linker_boundary_free():
    similarities = np.full((5, 5), -100.0, dtype=np.float32)
    similarities[:, (0, 4)] = 0.0
    similarities[0, 1] = 10.0
    similarities[2, 2] = 10.0
    similarities[4, 3] = 10.0
    lengths = np.asarray(similarities.shape)

    penalized = _affine_score(
        similarities,
        lengths,
        1e-4,
        gap_extend=-1.0,
        gap_open=-3.0,
    )
    free = _affine_score(
        similarities,
        lengths,
        1e-4,
        gap_extend=-1.0,
        gap_open=-3.0,
        free_gap_boundary=(1, 2),
    )

    assert float(penalized) == pytest.approx(24.0)
    assert float(free) == pytest.approx(30.0)


def test_softalign_mode_loads_its_complete_parameter_set():
    references = load_references(2.0, "softalign")
    assert tuple(references) == constants.CHAIN_TYPES
    for embeddings, positions in references.values():
        assert embeddings.shape == (len(positions), constants.EMBED_DIM)

    parameters = load_parameters("softalign")
    assert load_parameters("softalign") is parameters
    assert set(parameters) == set(load_parameters("sabr"))

    gap_extend, gap_open = load_gap_penalties("softalign")
    assert gap_extend == pytest.approx(0.1942468136548996)
    assert gap_open == pytest.approx(-2.5441808700561523)


def test_softalign_gap_penalties_reach_affine_alignment(monkeypatch):
    captured = {}

    def fake_affine(
        similarity,
        lengths,
        temperature,
        gap_extend,
        gap_open,
        free_gap_boundary,
        *,
        with_gradient,
    ):
        assert with_gradient
        captured["gap_extend"] = gap_extend
        captured["gap_open"] = gap_open
        captured["free_gap_boundary"] = free_gap_boundary
        return np.float32(0.0), np.zeros(np.asarray(similarity).shape)

    monkeypatch.setattr("sabr.alignment._affine_value_and_grad", fake_affine)
    _align_reference(
        np.zeros((1, constants.EMBED_DIM)),
        np.zeros((1, constants.EMBED_DIM)),
        "softalign",
    )
    assert captured == {
        "gap_extend": pytest.approx(0.1942468136548996),
        "gap_open": pytest.approx(-2.5441808700561523),
        "free_gap_boundary": -1,
    }


def test_model_assets_are_cached_and_immutable():
    load_parameters.cache_clear()
    first_parameters = load_parameters()
    assert load_parameters() is first_parameters

    load_references.cache_clear()
    first_references = load_references(0.0)
    assert load_references(0.0) is first_references
    for embeddings, positions in first_references.values():
        assert not embeddings.flags.writeable
        assert isinstance(positions, tuple)


def test_auto_reference_ties_resolve_in_h_k_l_order(monkeypatch):
    references = {
        chain_type: (np.zeros((1, 64)), [1])
        for chain_type in constants.CHAIN_TYPES
    }
    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references,
    )
    monkeypatch.setattr(
        "sabr.alignment._align_reference",
        lambda query, reference, mode, score_only=False: (
            None if score_only else np.ones((len(query), 1)),
            np.zeros((len(query), 3)),
            1.0,
        ),
    )
    monkeypatch.setattr(
        "sabr.alignment.apply_corrections",
        lambda alignment, gap_indices: alignment,
    )
    _, selected, _ = align(np.zeros((1, 64)), frozenset(), "auto", 0.0)
    assert selected == "H"


@pytest.mark.parametrize(
    ("chain_type", "expected_calls"),
    (
        ("auto", [False, False, False, True]),
        ("H,K", [True, True]),
        ("H", [True]),
    ),
)
def test_reference_search_limits_gradient_work(
    monkeypatch, chain_type, expected_calls
):
    import sabr.alignment as alignment_module

    original = alignment_module._affine_value_and_grad
    gradient_calls = []

    def capture(*args, **kwargs):
        gradient_calls.append(kwargs.get("with_gradient", True))
        return original(*args, **kwargs)

    monkeypatch.setattr(alignment_module, "_affine_value_and_grad", capture)
    baseline = np.load(DATA / "math_baseline.npz")
    _, selected, _ = align(baseline["embeddings"], frozenset(), chain_type, 0.0)
    assert selected == "H"
    assert gradient_calls == expected_calls


def test_auto_search_rejects_non_finite_winning_gradient(monkeypatch):
    references = {
        chain_type: (np.zeros((1, 64)), (1,))
        for chain_type in constants.CHAIN_TYPES
    }
    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references,
    )
    monkeypatch.setattr(
        "sabr.alignment._align_reference",
        lambda query, reference, mode, score_only=False: (
            None if score_only else np.full((len(query), 1), np.nan),
            np.zeros((len(query), 3)),
            1.0,
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        align(np.zeros((1, 64)), frozenset(), "auto", 0.0)


@pytest.mark.parametrize("chain_type", constants.CHAIN_TYPES)
def test_explicit_chain_type_aligns_only_its_reference(monkeypatch, chain_type):
    references = {
        candidate: (np.full((1, 64), index), [1])
        for index, candidate in enumerate(constants.CHAIN_TYPES)
    }
    seen = []

    def fake_align(query, reference, mode):
        seen.append(int(reference[0, 0]))
        return np.ones((len(query), 1)), np.zeros((len(query), 3)), 1.0

    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references,
    )
    monkeypatch.setattr("sabr.alignment._align_reference", fake_align)
    monkeypatch.setattr(
        "sabr.alignment.apply_corrections",
        lambda alignment, gap_indices: alignment,
    )
    _, selected, _ = align(np.zeros((1, 64)), frozenset(), chain_type, 0.0)
    assert selected == chain_type
    assert seen == [constants.CHAIN_TYPES.index(chain_type)]


@pytest.mark.parametrize("chain_type", constants.TCR_CHAIN_TYPES)
def test_tcr_chain_type_aligns_only_kappa_reference(monkeypatch, chain_type):
    references = {
        candidate: (np.full((1, 64), index), [1])
        for index, candidate in enumerate(constants.CHAIN_TYPES)
    }
    seen = []

    def fake_align(query, reference, mode):
        seen.append(int(reference[0, 0]))
        return np.ones((len(query), 1)), np.zeros((len(query), 3)), 1.0

    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references,
    )
    monkeypatch.setattr("sabr.alignment._align_reference", fake_align)
    monkeypatch.setattr(
        "sabr.alignment.apply_corrections",
        lambda alignment, gap_indices: alignment,
    )

    _, selected, _ = align(np.zeros((1, 64)), frozenset(), chain_type, 0.0)

    assert selected == "K"
    assert seen == [constants.CHAIN_TYPES.index("K")]


def test_scfv_mode_selects_and_corrects_a_composite_reference(monkeypatch):
    representations = (
        *constants.CHAIN_TYPES,
        *constants.SCFV_CHAIN_TYPES,
    )
    references = {}
    for index, representation in enumerate(representations):
        domain_count = len(representation)
        references[representation] = (
            np.full((domain_count, 64), index),
            [1, 129] if domain_count == 2 else [1],
        )

    seen_boundaries = {}

    def fake_align(query, reference, mode, free_gap_boundary=-1):
        reduced = np.zeros((len(query), len(reference)))
        np.fill_diagonal(reduced, 1)
        marker = int(reference[0, 0])
        representation = representations[marker]
        seen_boundaries[representation] = free_gap_boundary
        if representation == "H":
            score = 100.0
        else:
            score = 10.0 if representation == "HK" else 1.0
        return reduced, np.zeros((len(query), len(reference) + 2)), score

    corrected_shapes = []
    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references if scfv else None,
    )
    monkeypatch.setattr("sabr.alignment._align_reference", fake_align)
    monkeypatch.setattr(
        "sabr.alignment.apply_corrections",
        lambda alignment, gap_indices: (
            corrected_shapes.append(alignment.shape) or alignment
        ),
    )

    alignment, selected, score = align(
        np.zeros((2, 64)), frozenset(), "auto", 0.0, scfv=True
    )
    assert selected == "HK"
    assert score == 10.0
    assert alignment.shape == (2, 256)
    assert corrected_shapes == [(2, 128), (2, 128)]
    assert seen_boundaries == {
        "HK": (1,),
        "HL": (1,),
        "KH": (1,),
        "LH": (1,),
    }


def test_requested_candidates_and_higher_order_boundaries_are_exact(
    monkeypatch,
):
    references = {
        "H": (np.zeros((1, 64)), (1,)),
        "K": (np.ones((1, 64)), (1,)),
        "L": (np.full((1, 64), 2), (1,)),
    }
    seen = []

    def fake_align(query, reference, mode, free_gap_boundary=-1):
        seen.append((int(reference[0, 0]), free_gap_boundary))
        reduced = np.zeros((len(query), len(reference)))
        np.fill_diagonal(reduced, 1)
        return reduced, np.zeros((len(query), len(reference) + 2)), 1.0

    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references,
    )
    monkeypatch.setattr("sabr.alignment._align_reference", fake_align)
    monkeypatch.setattr(
        "sabr.alignment.apply_corrections",
        lambda alignment, gap_indices: alignment,
    )

    alignment, selected, _ = align(
        np.zeros((3, 64)), frozenset(), "HHK,HHL", 0.0
    )
    assert selected == "HHK"
    assert alignment.shape == (3, 3 * constants.IMGT_MAX_POSITION)
    assert seen == [(0, (1, 2)), (0, (1, 2))]


def test_terminal_gap_penalty_uses_normal_affine_gap_costs():
    alignment = np.zeros((8, 10))
    alignment[2, 3] = 1
    alignment[5, 7] = 1
    expected = 4 * constants.SW_GAP_OPEN + 5 * constants.SW_GAP_EXTEND
    assert _affine_gap_penalty(0) == 0.0
    assert _affine_gap_penalty(1) == constants.SW_GAP_OPEN
    assert _terminal_gap_penalty(alignment) == pytest.approx(expected)
    soft_extend, soft_open = load_gap_penalties("softalign")
    assert _terminal_gap_penalty(alignment, "softalign") == pytest.approx(
        4 * soft_open + 5 * soft_extend
    )


def test_multidomain_terminal_penalty_is_used_only_for_candidate_selection(
    monkeypatch,
):
    references = {
        "H": (np.zeros((1, 64)), [1]),
        "HK": (np.ones((4, 64)), [1, 2, 129, 130]),
    }

    def fake_align(query, reference, mode, free_gap_boundary=-1):
        if len(reference) == 1:
            return np.ones((1, 1)), np.zeros((1, 3)), 8.0
        return (
            np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            np.zeros((1, 6)),
            10.0,
        )

    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: references,
    )
    monkeypatch.setattr("sabr.alignment._align_reference", fake_align)
    monkeypatch.setattr(
        "sabr.alignment.apply_corrections",
        lambda alignment, gap_indices: alignment,
    )

    alignment, selected, score = align(
        np.zeros((1, 64)), frozenset(), "H,HK", 0.0
    )
    assert selected == "H"
    assert score == 8.0
    assert alignment.shape == (1, constants.IMGT_MAX_POSITION)


def test_huge_internal_run_is_valid_inside_one_cdr():
    alignment = np.zeros((102, 128), dtype=int)
    alignment[0, 26] = 1
    alignment[101, 37] = 1
    _alignment_path(alignment)


def test_de_loop_insertions_form_a_valid_alignment():
    alignment = np.zeros((8, 128), dtype=int)
    alignment[0, 78] = 1
    alignment[1, 79] = 1
    alignment[2, 80] = 1
    alignment[3, 81] = 1
    alignment[5, 82] = 1
    alignment[6, 83] = 1
    alignment[7, 84] = 1
    _alignment_path(alignment)


@pytest.mark.parametrize(
    ("alignment", "message"),
    [
        (np.asarray([[0.0, np.nan]]), "finite"),
        (np.asarray([[0, 2]]), "zeroes and ones"),
        (np.asarray([[1, 1], [0, 0]]), "query row"),
        (np.asarray([[1, 0], [1, 0]]), "IMGT column"),
        (np.asarray([[0, 1], [1, 0]]), "monotonic"),
    ],
)
def test_malformed_alignments_are_rejected(alignment, message):
    with pytest.raises(ValueError, match=message):
        _alignment_path(alignment)


def test_internal_framework_run_is_allowed():
    alignment = np.zeros((4, 128), dtype=int)
    alignment[0, 19] = 1
    alignment[3, 20] = 1
    _alignment_path(alignment)
    states, _, _ = alignment_to_states(alignment)
    assert [state for state, _ in states] == [
        (20, "m"),
        (20, "i"),
        (20, "i"),
        (21, "m"),
    ]


def test_leading_and_trailing_alignment_boundaries_are_allowed():
    for first_row in (2, 3, 4):
        leading = np.zeros((first_row + 1, 128), dtype=int)
        leading[first_row, 2] = 1
        _alignment_path(leading)

    for last_position in (123, 124, 125):
        trailing = np.zeros((3, 128), dtype=int)
        trailing[0, last_position - 1] = 1
        _alignment_path(trailing)


def test_non_finite_reference_score_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: {"H": (np.zeros((1, 64)), (1,))},
    )
    monkeypatch.setattr(
        "sabr.alignment._align_reference",
        lambda query, reference, mode: (
            np.ones((len(query), 1)),
            np.zeros((len(query), 1)),
            float("nan"),
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        align(np.zeros((1, 64)), frozenset(), "H", 0.0)


def test_non_finite_similarity_matrix_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "sabr.alignment.load_references",
        lambda noise, mode, scfv=False: {"H": (np.zeros((1, 64)), (1,))},
    )
    monkeypatch.setattr(
        "sabr.alignment._align_reference",
        lambda query, reference, mode: (
            np.ones((len(query), 1)),
            np.full((len(query), 1), np.inf),
            1.0,
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        align(np.zeros((1, 64)), frozenset(), "H", 0.0)
