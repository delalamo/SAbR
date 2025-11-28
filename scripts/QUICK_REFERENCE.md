# Quick Reference - SAbDab Processing Pipeline

## TL;DR - Most Common Commands

```bash
# First time run (full pipeline)
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data

# Resume after interruption
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data

# Test with one category only
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data \
    --categories human_heavy_kinked \
    --skip-embeddings
```

## Expected Runtime

| Step | Time | Notes |
|------|------|-------|
| 1. Download summary | 1 min | Network required |
| 2. Download structures | 2-6 hrs | 8.2 GB, network required |
| 3. Organize | 10-30 min | Single-threaded |
| 4. Classify CDR-H3 | 5-15 min | Single-threaded |
| 5. Cluster sequences | 30-60 min | Multi-threaded |
| 6. Compute embeddings | 30 min - 48 hrs | **GPU: 30-120 min, CPU: 10-48 hrs** |
| **TOTAL** | **8-48 hrs** | **GPU recommended** |

## Resource Requirements

- **Storage**: 15-20 GB
- **Memory**: 16 GB RAM (4 GB minimum)
- **GPU**: 4-8 GB VRAM (optional but **highly recommended**)
- **Network**: Required for steps 1-2

## Common Scenarios

### Scenario 1: First Time Setup
```bash
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data
```

### Scenario 2: Job Got Interrupted
```bash
# Just run the same command - it auto-resumes
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data
```

### Scenario 3: Already Have Structures Downloaded
```bash
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data \
    --skip-download
```

### Scenario 4: Only Process Human Antibodies
```bash
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data \
    --categories human_heavy_kinked,human_heavy_extended,human_light_all
```

### Scenario 5: Run in Background (Long Jobs)
```bash
nohup python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data \
    > pipeline.log 2>&1 &

# Monitor with:
tail -f pipeline.log
```

### Scenario 6: Testing the Pipeline
```bash
# Quick test with smallest category
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/test_run \
    --categories llama_alpaca_light_all \
    --skip-embeddings
```

## Output Structure

```
output_dir/
├── .pipeline_state.json              # Resume state (auto-generated)
├── summary.tsv                        # SAbDab metadata
├── all_structures/imgt/               # Downloaded PDBs (8.2 GB)
├── organized_antibodies/
│   ├── cluster_representatives.csv   # Main output (1,433 clusters)
│   └── {species}/{chain}/{class}/    # Organized PDB files
└── mpnn_embeddings/
    └── {category}_embeddings.npz     # Per-residue embeddings
```

## Key Files

| File | Description | Size |
|------|-------------|------|
| `cluster_representatives.csv` | Main output: cluster info | ~400 KB |
| `{category}_embeddings.npz` | MPNN embeddings per category | 100 KB - 11 MB |
| `.pipeline_state.json` | Resume state (auto-created) | ~10 KB |

## Monitoring Progress

### Check Current Status
```bash
# View log in real-time
tail -f ~/antibody_data/pipeline.log

# Check state file
cat ~/antibody_data/.pipeline_state.json

# Check which steps are done
grep "completed_steps" ~/antibody_data/.pipeline_state.json
```

### Check Disk Usage
```bash
du -sh ~/antibody_data/*
```

### Check Output Files
```bash
# Count structures downloaded
ls ~/antibody_data/all_structures/imgt/*.pdb | wc -l

# Count cluster representatives
wc -l ~/antibody_data/organized_antibodies/cluster_representatives.csv

# List embedding files
ls -lh ~/antibody_data/mpnn_embeddings/
```

## Troubleshooting

### Out of Memory
```bash
# Close other applications
# Or use smaller categories:
--categories human_heavy_kinked
```

### Network Timeout
```bash
# Just resume - downloaded files are skipped
python scripts/process_sabdab_antibodies.py \
    --output-dir ~/antibody_data
```

### GPU Out of Memory
```bash
# Will automatically fall back to CPU
# Or process one category at a time:
--categories human_heavy_kinked
```

### MMSeqs2 Not Found
```bash
conda install -c conda-forge -c bioconda mmseqs2
```

### Script Won't Run
```bash
# Check you're in the right environment
conda activate softalign_env

# Or use conda run
python scripts/process_sabdab_antibodies.py --help
```

## All Available Options

```
--output-dir DIR          Base output directory (REQUIRED)
--skip-download           Skip downloading structures
--skip-clustering         Skip clustering step
--skip-embeddings         Skip MPNN computation
--min-seq-id FLOAT        Clustering identity (default: 0.7)
--save-interval INT       Save every N structures (default: 10)
--categories CATS         Process specific categories only
--resume                  Resume from previous run (auto-detected)
```

## Available Categories

**Heavy chains (kinked/extended/unclassified):**
- `human_heavy_kinked`
- `human_heavy_extended`
- `mouse_heavy_kinked`
- `mouse_heavy_extended`
- `llama_alpaca_heavy_kinked`
- `llama_alpaca_heavy_extended`
- `other_heavy_kinked`
- `other_heavy_extended`

**Light chains (all):**
- `human_light_all`
- `mouse_light_all`
- `llama_alpaca_light_all`
- `other_light_all`

## Performance Tips

1. **Use GPU** - 10-100x faster for embeddings
2. **Use SSD** - Faster for I/O intensive operations
3. **Stable network** - For downloading structures
4. **Close other apps** - Free up RAM
5. **Process specific categories** - Faster for testing

## Getting Help

```bash
# Show full help
python scripts/process_sabdab_antibodies.py --help

# Interactive examples
bash scripts/example_usage.sh

# Check README
cat scripts/README.md
```
