import numpy as np
import pytest

from sabr.alignment.soft_aligner import SoftAlignOutput
from sabr.embeddings.mpnn import MPNNEmbeddings


def test_mpnnembeddings_shape_mismatch_raises():
    # embeddings has 2 rows, idxs has 3 items -> mismatch
    embedding = np.zeros((2, 5), dtype=float)
    idx = ["a", "b", "c"]

    with pytest.raises(ValueError) as excinfo:
        MPNNEmbeddings(name="test_case", embeddings=embedding, idxs=idx)

    # Check key parts of the error message
    msg = str(excinfo.value)
    assert "embeddings.shape[0] (2) must match len(idxs) (3)" in msg
    assert "Error raised for test_case" in msg


def test_softalignoutput_holds_passed_values():
    alignment = np.ones((2, 2), dtype=int)
    output = SoftAlignOutput(
        alignment=alignment,
        score=1.5,
        sim_matrix=None,
        chain_type="mouse",
        idxs1=[str(x) for x in range(len(alignment[0]))],
        idxs2=[str(x) for x in range(len(alignment[1]))],
    )

    assert output.alignment.shape == (2, 2)
    assert output.score == pytest.approx(1.5)
    assert output.chain_type == "mouse"
