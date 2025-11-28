# SAbR Scripts

This directory contains scripts for processing antibody structures and computing embeddings.

## process_sabdab_antibodies.py

Complete pipeline for downloading and processing antibody structures from the SAbDab database.

### Quick Start

```bash
# Full pipeline
python scripts/process_sabdab_antibodies.py --output-dir ~/antibody_data

# Resume from interruption
python scripts/process_sabdab_antibodies.py --output-dir ~/antibody_data

# Skip downloading structures (if already downloaded)
python scripts/process_sabdab_antibodies.py --output-dir ~/antibody_data --skip-download
```

### Features

- ✅ **Resume capability**: Automatically resumes from interruption
- ✅ **Error handling**: Try/except blocks around all file operations
- ✅ **Incremental saving**: Progress saved every 10 structures
- ✅ **State tracking**: JSON state file tracks all progress
- ✅ **Category filtering**: Process specific species/chain combinations
- ✅ **Detailed logging**: Comprehensive logging with progress bars

### Pipeline Steps

1. **Download SAbDab summary** (~1 min)
2. **Download IMGT structures** (2-6 hours, 8.2 GB)
3. **Organize by species/chain** (10-30 min, 2.4 GB)
4. **Classify CDR-H3 conformations** (5-15 min)
5. **Cluster at 70% identity** (30-60 min)
6. **Compute MPNN embeddings** (30 min - 48 hours)

### Resource Requirements

- **Total time**: 8-48 hours (GPU: 2-8 hours, CPU: 10-48 hours)
- **Storage**: 15-20 GB
- **Memory**: 16 GB recommended (4 GB minimum)
- **GPU**: 4-8 GB VRAM (optional but highly recommended)

### Options

```
--output-dir DIR        Base output directory (required)
--skip-download         Skip downloading structures
--skip-clustering       Skip clustering step
--skip-embeddings       Skip MPNN computation
--min-seq-id FLOAT      Clustering identity (default: 0.7)
--save-interval INT     Save every N structures (default: 10)
--categories CATS       Filter categories (comma-separated)
```

### Examples

```bash
# Process only human heavy kinked
python scripts/process_sabdab_antibodies.py \\
    --output-dir ~/antibody_data \\
    --categories human_heavy_kinked

# Process multiple categories
python scripts/process_sabdab_antibodies.py \\
    --output-dir ~/antibody_data \\
    --categories human_heavy_kinked,human_heavy_extended,human_light_all

# Skip embedding computation (run only data processing)
python scripts/process_sabdab_antibodies.py \\
    --output-dir ~/antibody_data \\
    --skip-embeddings
```

### Output Structure

```
output_dir/
├── .pipeline_state.json          # Resume state
├── summary.tsv                    # SAbDab metadata
├── all_structures/
│   └── imgt/                      # IMGT-numbered PDBs (~8.2 GB)
├── organized_antibodies/
│   ├── cluster_representatives.csv
│   ├── human/
│   │   ├── heavy/
│   │   │   ├── kinked/           # Kinked CDR-H3
│   │   │   ├── extended/         # Extended CDR-H3
│   │   │   └── unclassified/     # Missing residues
│   │   └── light/
│   │       └── all/
│   ├── mouse/
│   ├── llama_alpaca/
│   └── other/
└── mpnn_embeddings/
    ├── human_heavy_kinked_embeddings.npz
    ├── human_heavy_extended_embeddings.npz
    ├── human_light_all_embeddings.npz
    └── ...
```

### Dependencies

```bash
# Core dependencies
pip install numpy pandas biopython tqdm

# JAX/Haiku (for embeddings)
pip install jax haiku dm-haiku

# MMSeqs2 (for clustering)
conda install -c conda-forge -c bioconda mmseqs2
```

### Resume Capability

The script automatically saves progress to `.pipeline_state.json` including:
- Completed pipeline steps
- Failed files with error messages
- Number of structures processed
- Last processed category

If interrupted, simply run the same command again and it will resume from where it left off.

### Error Handling

The script includes comprehensive error handling:
- **Network failures**: Retries and continues with remaining files
- **Invalid PDB files**: Skipped and logged
- **Missing residues**: Moved to unclassified category
- **Computation failures**: Logged and skipped, processing continues

All failures are recorded in the state file for review.

### Performance Tips

1. **Use GPU**: Step 6 is 10-100x faster with GPU
2. **Use SSD**: I/O intensive operations benefit from fast storage
3. **Network**: Stable connection needed for steps 1-2
4. **Memory**: Close other applications if RAM is limited
5. **Categories**: Process specific categories to reduce time

### Troubleshooting

**Out of memory during clustering:**
```bash
# Reduce coverage parameter (edit script, line ~1177)
# Change "-c", "0.8" to "-c", "0.5"
```

**GPU out of memory:**
```bash
# Reduce batch size or use CPU
# JAX will automatically fall back to CPU if GPU unavailable
```

**Network timeouts:**
```bash
# Resume after network is stable
# Already downloaded files are skipped
```

**Missing MMSeqs2:**
```bash
conda install -c conda-forge -c bioconda mmseqs2
```

### Citation

If you use this pipeline, please cite:
- SAbDab: Dunbar et al. (2014) Nucleic Acids Research
- SAbR: [Your citation]
- MPNN: [Original MPNN paper]
