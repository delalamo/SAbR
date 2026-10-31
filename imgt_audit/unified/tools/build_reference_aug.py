"""Train the shipped unified reference on all annotated domains and write the SAbR asset."""
import sys, time
import numpy as np
sys.path.insert(0, "/tmp/sabr_unified/tools")
import ulib

L2, SCALE = 1e-6, 3.0
OUT = "/tmp/sabr_unified/unified_reference_aug.npz"

if __name__ == "__main__":
    t0 = time.time()
    doms, cache, miss = ulib.load_dataset()
    names, ref, info = ulib.train_profile(doms, cache, ulib.anchor_list(), l2=L2, maxiter=2000,
                                          negatives=ulib.load_negatives(), neg_weight=1.0, augment=True)
    order = sorted(range(len(names)), key=lambda k: ulib.lab_key(names[k]))
    labels = np.array([names[k] for k in order])
    emb = (ref[order] * SCALE).astype(np.float32)
    np.savez_compressed(OUT, labels=labels, embeddings=emb)
    print("domains", len(doms), "anchors", len(labels), info, f"{time.time() - t0:.0f}s")
