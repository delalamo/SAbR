"""NumPy inference for the saved ProteinMPNN/SoftAlign encoders.

SciPy supplies the exact erf-based GELU. Training, stochastic augmentation,
JAX tracing, and Haiku module construction are unnecessary for inference.
"""

import functools
from importlib.resources import files

import numpy as np
from scipy.special import erfc

from sabr import constants
from sabr._types import FloatArray, IntArray


def _linear(x: FloatArray, parameters: dict[str, FloatArray]) -> FloatArray:
    """Map [..., input width] to [..., output width] with saved weights."""
    result = x @ parameters["w"]
    if "b" in parameters:
        result += parameters["b"]
    return result


def _norm(x: FloatArray, parameters: dict[str, FloatArray]) -> FloatArray:
    """Normalize the final feature axis, preserving all dimensions of x."""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True, mean=mean)
    inv = parameters["scale"] * np.reciprocal(np.sqrt(variance + 1e-5))
    return inv * (x - mean) + parameters["offset"]


def _gelu(x: FloatArray) -> FloatArray:
    """Apply the trained exact GELU elementwise without changing shape."""
    return (0.5 * x) * erfc(-x * np.float32(2**-0.5))


def _rbf(distances: FloatArray) -> FloatArray:
    """Expand Angstrom distances [...] to [..., 16] radial basis features."""
    centers = np.linspace(2, 22, 16, dtype=np.float32)
    return np.exp(-(((distances[..., None] - centers) / 1.25) ** 2))


def _features(
    coords: FloatArray, parameters: dict[str, dict[str, FloatArray]]
) -> tuple[FloatArray, IntArray]:
    """Map [N, 4, 3] coordinates to [N, K, 64] edges and [N, K] neighbors.

    N is the residue count and K = min(N, 64), including the central residue.
    The pairwise CA distance matrix is [N, N]; subsequent atom-pair features
    use only [N, K] distances. The 25 radial blocks each have 16 channels.
    The fourth coordinate is the historical synthetic CB input, although the
    trained feature layout calls it oxygen. See ``encode`` for this contract.
    """
    n, ca, c, oxygen = np.moveaxis(coords, 1, 0)
    b = ca - n
    d = c - ca
    cb = -0.58273431 * np.cross(b, d) + 0.56802827 * b - 0.54067466 * d + ca
    distances = np.sqrt(
        np.sum((ca[:, None] - ca[None, :]) ** 2, axis=-1) + 1e-6
    )
    # Stable sorting preserves lower-index precedence when distances tie.
    neighbors = np.argsort(distances, axis=-1, kind="stable")[:, :64]
    # Fill the final feature matrix without retaining every radial block.
    positional = parameters[
        "protein_features/~/positional_encodings/~/embedding_linear"
    ]
    positional_width = positional["b"].shape[0]
    edges = np.empty(
        (*neighbors.shape, positional_width + 25 * 16), dtype=np.float32
    )
    # Preserve the historical argument inversion: residue indices were
    # chain labels and all actual positional offsets were zero.
    positions = np.where(neighbors == np.arange(len(coords))[:, None], 32, 65)
    edges[..., :positional_width] = positional["w"][positions] + positional["b"]
    edges[..., positional_width : positional_width + 16] = _rbf(
        np.take_along_axis(distances, neighbors, axis=1)
    )
    n_neighbors, ca_neighbors, c_neighbors, oxygen_neighbors, cb_neighbors = (
        atom[neighbors] for atom in (n, ca, c, oxygen, cb)
    )
    pairs = (
        (n, n_neighbors),
        (c, c_neighbors),
        (oxygen, oxygen_neighbors),
        (cb, cb_neighbors),
        (ca, n_neighbors),
        (ca, c_neighbors),
        (ca, oxygen_neighbors),
        (ca, cb_neighbors),
        (n, c_neighbors),
        (n, oxygen_neighbors),
        (n, cb_neighbors),
        (cb, c_neighbors),
        (cb, oxygen_neighbors),
        (oxygen, c_neighbors),
        (n, ca_neighbors),
        (c, ca_neighbors),
        (oxygen, ca_neighbors),
        (cb, ca_neighbors),
        (c, n_neighbors),
        (oxygen, n_neighbors),
        (cb, n_neighbors),
        (c, cb_neighbors),
        (oxygen, cb_neighbors),
        (c, oxygen_neighbors),
    )
    # Compute only selected neighbor distances instead of 24 dense N x N
    # distance matrices. Atom-pair order is part of the trained model.
    for index, (first, second_neighbors) in enumerate(pairs, start=1):
        distance = np.sqrt(
            np.sum((first[:, None] - second_neighbors) ** 2, axis=-1) + 1e-6
        )
        start = positional_width + index * 16
        edges[..., start : start + 16] = _rbf(distance)
    edges = _linear(edges, parameters["protein_features/~/edge_embedding"])
    return _norm(edges, parameters["protein_features/~/norm_edges"]), neighbors


def _messages(
    nodes: FloatArray,
    edges: FloatArray,
    neighbors: IntArray,
    parameters: dict[str, dict[str, FloatArray]],
    prefix: str,
    names: tuple[str, str, str],
) -> FloatArray:
    """Return [N, K, H] messages for [N, H] nodes and [N, K, H] edges.

    ``neighbors`` is [N, K]; concatenated center/edge/neighbor features are
    [N, K, 3 * H]. Here H is the saved encoder's 64-channel hidden width.
    """
    center = np.broadcast_to(nodes[:, None], edges.shape)
    joined = np.concatenate((center, edges, nodes[neighbors]), axis=-1)
    first, second, third = (parameters[prefix + name] for name in names)
    return _linear(_gelu(_linear(_gelu(_linear(joined, first)), second)), third)


@functools.cache
def load_parameters(mode: str = "sabr") -> dict[str, dict[str, FloatArray]]:
    """Load the immutable encoder parameters for one alignment mode."""
    filenames = {
        "sabr": "mpnn_encoder.npz",
        "softalign": "softalign_encoder.npz",
    }
    try:
        filename = filenames[mode]
    except KeyError as error:
        raise ValueError(f"mode must be one of {constants.MODES}.") from error
    path = files("sabr.assets") / filename
    parameters = {}
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        for key in data.files:
            module, name = key.rsplit(".", 1)
            value = data[key]
            value.flags.writeable = False
            parameters.setdefault(module, {})[name] = value
    return parameters


def encode(coords: FloatArray, mode: str = "sabr") -> FloatArray:
    """Map finite [N, 4, 3] coordinates to float32 [N, 64] embeddings.

    N must be nonzero. Coordinates are in Angstroms and ordered N, CA, C,
    synthetic CB as emitted by ``extract_chain``. The saved model consumes
    that fourth slot as its historical oxygen feature and also computes its
    own virtual CB. This convention and the historical positional encoding
    are part of the trained model contract; neither is corrected at inference.
    """
    parameters = load_parameters(mode)
    coords = np.asarray(coords, dtype=np.float32)
    edges, neighbors = _features(coords, parameters)
    nodes = np.zeros((len(coords), constants.EMBED_DIM), dtype=np.float32)
    edges = _linear(edges, parameters["W_e"])
    for layer in range(constants.N_MPNN_LAYERS):
        module = "enc_layer" + (f"_{layer}" if layer else "") + "/~/"
        prefix = module + f"enc{layer}_"
        messages = _messages(
            nodes, edges, neighbors, parameters, prefix, ("W1", "W2", "W3")
        )
        nodes = _norm(
            nodes + np.sum(messages, axis=-2) / 30, parameters[prefix + "norm1"]
        )
        dense = module + f"position_wise_feed_forward/~/enc{layer}_dense_"
        hidden = _gelu(_linear(nodes, parameters[dense + "W_in"]))
        nodes = _norm(
            nodes + _linear(hidden, parameters[dense + "W_out"]),
            parameters[prefix + "norm2"],
        )
        # The last edge update cannot affect the returned node embeddings.
        if layer + 1 < constants.N_MPNN_LAYERS:
            messages = _messages(
                nodes,
                edges,
                neighbors,
                parameters,
                prefix,
                ("W11", "W12", "W13"),
            )
            edges = _norm(edges + messages, parameters[prefix + "norm3"])
    return nodes
