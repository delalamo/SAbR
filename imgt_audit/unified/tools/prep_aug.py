"""Noise-augmented embeddings: each chain re-encoded once with Gaussian backbone noise (0.25 or 0.5 A)."""
import glob, hashlib, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/home/user/SAbR/src")
from sabr.model import encode


def job(path):
    out = path.replace("/emb/", "/emb_aug/")
    if os.path.exists(out):
        return
    c = np.load(path)
    h = int(hashlib.sha1(os.path.basename(path).encode()).hexdigest(), 16)
    sigma = (0.25, 0.5)[h % 2]
    rng = np.random.default_rng(h % (2 ** 32))
    bb = c["coords"][:, :3].astype(np.float64) + rng.normal(0, sigma, c["coords"][:, :3].shape)
    n, ca, cc = bb[:, 0], bb[:, 1], bb[:, 2]
    b, e = ca - n, cc - ca
    cb = -0.58273431 * np.cross(b, e) + 0.56802827 * b - 0.54067466 * e + ca
    E = encode(np.concatenate([bb, cb[:, None]], 1).astype(np.float32))
    np.savez_compressed(out, emb=E.astype(np.float32), sigma=sigma)


if __name__ == "__main__":
    files = sorted(glob.glob("/tmp/sabr_unified/emb/*.npz"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        p.map(job, files, chunksize=4)
    print("done", len(glob.glob("/tmp/sabr_unified/emb_aug/*.npz")))
