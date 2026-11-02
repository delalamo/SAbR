# Commands used for the slow-predictor timing runs (6BPE, CPU, 4 cores)
boltz predict in_boltz/6bpe_BC.yaml --accelerator cpu --use_msa_server --cache boltz_cache --out_dir out_boltz --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1
colabfold_batch in_cf/6bpe_BC.fasta out_cf/6bpe_BC --data cf_data --model-type alphafold2_multimer_v3 --num-models 1 --num-recycle 3 --msa-mode mmseqs2_uniref_env
python h19_esmfold.py cases.json out/esmfold 6bpe_BC
