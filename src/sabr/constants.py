from softalign import END_TO_END_MODELS

EMBED_DIM = 64
N_MPNN_LAYERS = 3

E2E_MODEL = END_TO_END_MODELS.END_TO_END(
    EMBED_DIM,
    EMBED_DIM,
    EMBED_DIM,
    N_MPNN_LAYERS,
    EMBED_DIM,
    affine=True,
    soft_max=False,
    dropout=0.0,
    augment_eps=0.0,
)
