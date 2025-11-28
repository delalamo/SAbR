#!/usr/bin/env python3
"""
SAbDab Antibody Processing Pipeline

This script downloads antibody structures from SAbDab, organizes them by species and
chain type, classifies CDR-H3 conformations, clusters sequences, and computes MPNN
embeddings for structural analysis.

===============================================================================
EXPECTED RUNTIME & RESOURCE REQUIREMENTS
===============================================================================

Total Time: 8-48 hours (depends on network speed and GPU availability)

Step-by-step breakdown:
  1. Download SAbDab summary       ~1 minute      Network required
  2. Download structures (IMGT)    2-6 hours      Network required, 8.2 GB
  3. Organize by species/chain     10-30 minutes  Single-threaded, 2.4 GB output
  4. Classify CDR-H3 kinks         5-15 minutes   Single-threaded
  5. Cluster sequences (70%)       30-60 minutes  Multi-threaded, 4-8 GB RAM
  6. Compute MPNN embeddings       30 min-48 hrs  GPU highly recommended

Resource Requirements:
  - Storage: 15-20 GB total
    * Downloaded structures: 8.2 GB
    * Organized antibodies: 2.4 GB
    * MPNN embeddings: 3-7 GB
  - Memory: 16 GB RAM recommended (4 GB minimum for steps 1-4)
  - GPU: 4-8 GB VRAM (optional but highly recommended for step 6)
    * With GPU: Step 6 takes 30-120 minutes
    * Without GPU: Step 6 takes 10-48 hours
  - Network: Required for steps 1-2
  - CPU: 4+ cores recommended for clustering

Output:
  - Organized structures by species/chain/classification
  - Cluster representatives at 70% sequence identity
  - Per-residue MPNN embeddings for all cluster representatives

Resume Capability:
  - Each step checks for existing outputs and resumes from where it left off
  - Safe to interrupt and restart at any point
  - Progress is saved incrementally

Usage:
  python process_sabdab_antibodies.py --output-dir /path/to/output [options]

  Options:
    --output-dir DIR        Base output directory (required)
    --skip-download         Skip downloading structures if already present
    --skip-clustering       Skip clustering step if already done
    --skip-embeddings       Skip MPNN embedding computation
    --min-seq-id FLOAT      Clustering sequence identity (default: 0.7)
    --save-interval INT     Save embeddings every N structures (default: 10)
    --categories CATS       Process only specific categories (comma-separated)

Examples:
  # Full pipeline
  python process_sabdab_antibodies.py --output-dir ~/antibody_analysis

  # Resume after interruption
  python process_sabdab_antibodies.py --output-dir ~/antibody_analysis

  # Skip downloading (if structures already exist)
  python process_sabdab_antibodies.py --output-dir ~/antibody_analysis --skip-download

  # Process only human heavy kinked
  python process_sabdab_antibodies.py --output-dir ~/antibody_analysis \\
      --categories human_heavy_kinked

===============================================================================
"""

import sys
import os
import argparse
import logging
import urllib.request
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

# BioPython imports
try:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
except ImportError:
    print("ERROR: BioPython not installed. Install with: pip install biopython")
    sys.exit(1)

# JAX/Haiku imports (lazy loaded in step 6)
HAIKU_AVAILABLE = False
JAX_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


class PipelineState:
    """Tracks pipeline progress and enables resume capability."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from file or create new state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                LOGGER.warning(f"Could not load state file: {e}. Starting fresh.")

        return {
            'completed_steps': [],
            'failed_files': {},
            'last_category': None,
            'downloaded_structures': 0,
            'processed_chains': 0,
            'classified_chains': 0,
            'clusters_created': 0,
            'embeddings_computed': 0,
        }

    def save(self):
        """Save current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Failed to save state: {e}")

    def mark_step_complete(self, step: str):
        """Mark a step as completed."""
        if step not in self.state['completed_steps']:
            self.state['completed_steps'].append(step)
        self.save()

    def is_step_complete(self, step: str) -> bool:
        """Check if a step is already completed."""
        return step in self.state['completed_steps']

    def add_failed_file(self, step: str, file_path: str, error: str):
        """Record a failed file."""
        if step not in self.state['failed_files']:
            self.state['failed_files'][step] = []
        self.state['failed_files'][step].append({
            'file': file_path,
            'error': str(error)
        })
        self.save()


# ============================================================================
# STEP 1: Download SAbDab Summary
# ============================================================================

def download_summary(output_dir: Path, state: PipelineState) -> Path:
    """
    Download SAbDab summary TSV file.

    Expected time: ~1 minute
    Network: Required
    """
    LOGGER.info("="*70)
    LOGGER.info("STEP 1: Downloading SAbDab Summary")
    LOGGER.info("="*70)

    summary_file = output_dir / 'summary.tsv'

    if summary_file.exists() and state.is_step_complete('download_summary'):
        LOGGER.info(f"Summary file already exists: {summary_file}")
        return summary_file

    url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"

    try:
        LOGGER.info(f"Downloading from {url}")
        urllib.request.urlretrieve(url, summary_file)

        # Verify download
        if not summary_file.exists():
            raise FileNotFoundError("Download failed - file not created")

        file_size = summary_file.stat().st_size
        if file_size < 1000:  # Should be several MB
            raise ValueError(f"Downloaded file too small ({file_size} bytes)")

        LOGGER.info(f"✓ Summary downloaded: {summary_file} ({file_size:,} bytes)")
        state.mark_step_complete('download_summary')
        return summary_file

    except Exception as e:
        LOGGER.error(f"Failed to download summary: {e}")
        raise


# ============================================================================
# STEP 2: Download IMGT Structures
# ============================================================================

def download_imgt_structure(pdb_id: str, output_dir: Path) -> bool:
    """Download a single IMGT-numbered structure from SAbDab."""
    output_file = output_dir / f"{pdb_id}.pdb"

    if output_file.exists():
        return True  # Already downloaded

    url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/structurefile/{pdb_id}/imgt"

    try:
        urllib.request.urlretrieve(url, output_file)

        # Verify it's a valid PDB file
        if output_file.exists():
            with open(output_file, 'r') as f:
                content = f.read(500)  # Check first 500 chars
                if 'ATOM' in content or 'HETATM' in content:
                    return True

        # Invalid file - remove it
        if output_file.exists():
            output_file.unlink()
        return False

    except Exception as e:
        LOGGER.debug(f"Failed to download {pdb_id}: {e}")
        if output_file.exists():
            output_file.unlink()
        return False


def download_structures(summary_file: Path, output_dir: Path, state: PipelineState) -> Path:
    """
    Download all IMGT-numbered structures from SAbDab.

    Expected time: 2-6 hours (network dependent)
    Storage: ~8.2 GB
    Network: Required
    """
    LOGGER.info("="*70)
    LOGGER.info("STEP 2: Downloading IMGT Structures")
    LOGGER.info("="*70)

    structures_dir = output_dir / 'all_structures' / 'imgt'
    structures_dir.mkdir(parents=True, exist_ok=True)

    if state.is_step_complete('download_structures'):
        LOGGER.info(f"Structures already downloaded: {structures_dir}")
        return structures_dir

    # Read summary to get PDB IDs
    try:
        df = pd.read_csv(summary_file, sep='\t')
        pdb_ids = df['pdb'].str.lower().unique()
        LOGGER.info(f"Found {len(pdb_ids)} unique PDB IDs")
    except Exception as e:
        LOGGER.error(f"Failed to read summary file: {e}")
        raise

    # Download structures
    downloaded = 0
    failed = []

    # Check how many already exist
    existing = len(list(structures_dir.glob('*.pdb')))
    if existing > 0:
        LOGGER.info(f"Found {existing} existing structures, resuming download...")
        downloaded = existing

    for pdb_id in tqdm(pdb_ids, desc="Downloading structures"):
        try:
            if download_imgt_structure(pdb_id, structures_dir):
                downloaded += 1
                state.state['downloaded_structures'] = downloaded

                # Save state periodically
                if downloaded % 100 == 0:
                    state.save()
            else:
                failed.append(pdb_id)
                state.add_failed_file('download_structures', pdb_id, "Download failed")

        except KeyboardInterrupt:
            LOGGER.warning("Download interrupted by user")
            state.save()
            raise
        except Exception as e:
            LOGGER.warning(f"Error downloading {pdb_id}: {e}")
            failed.append(pdb_id)
            state.add_failed_file('download_structures', pdb_id, str(e))

    LOGGER.info(f"✓ Downloaded: {downloaded}/{len(pdb_ids)} structures")
    if failed:
        LOGGER.warning(f"Failed to download {len(failed)} structures")
        LOGGER.debug(f"Failed PDBs: {failed[:10]}...")  # Show first 10

    state.mark_step_complete('download_structures')
    return structures_dir


# ============================================================================
# STEP 3: Organize by Species and Chain
# ============================================================================

def normalize_species(species_str: str) -> str:
    """Map species to categories."""
    if not species_str or species_str == 'NA' or species_str == '':
        return 'other'

    species_lower = species_str.lower()

    if 'homo sapiens' in species_lower or 'human' in species_lower:
        return 'human'
    elif 'mus musculus' in species_lower or 'mouse' in species_lower:
        return 'mouse'
    elif 'lama' in species_lower or 'alpaca' in species_lower or 'vicugna' in species_lower:
        return 'llama_alpaca'
    else:
        return 'other'


def extract_chain_from_pdb(pdb_path: Path, chain_id: str, output_path: Path) -> bool:
    """Extract a specific chain from a PDB file."""
    if chain_id == 'NA' or not chain_id:
        return False

    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        chain_lines = []
        for line in lines:
            # Keep header lines
            if line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE', 'KEYWDS',
                               'EXPDTA', 'AUTHOR', 'REVDAT', 'JRNL', 'REMARK')):
                chain_lines.append(line)
            # Keep ATOM/HETATM lines for this chain
            elif line.startswith(('ATOM', 'HETATM')):
                if len(line) > 21 and line[21] == chain_id:
                    chain_lines.append(line)
            # Keep TER if it's for this chain
            elif line.startswith('TER'):
                if len(line) > 21 and line[21] == chain_id:
                    chain_lines.append(line)

        if not chain_lines:
            return False

        # Add END line
        chain_lines.append('END\n')

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.writelines(chain_lines)

        return True

    except Exception as e:
        LOGGER.debug(f"Error extracting chain {chain_id} from {pdb_path}: {e}")
        return False


def organize_antibodies(summary_file: Path, structures_dir: Path,
                        output_dir: Path, state: PipelineState) -> Path:
    """
    Filter scFvs and organize chains by species.

    Expected time: 10-30 minutes
    Storage: ~2.4 GB output
    CPU: Single-threaded
    """
    LOGGER.info("="*70)
    LOGGER.info("STEP 3: Organizing Antibodies by Species/Chain")
    LOGGER.info("="*70)

    organized_dir = output_dir / 'organized_antibodies'

    if state.is_step_complete('organize_antibodies'):
        LOGGER.info(f"Antibodies already organized: {organized_dir}")
        return organized_dir

    # Statistics
    stats = defaultdict(lambda: defaultdict(int))
    processed = 0
    skipped = 0

    try:
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in tqdm(reader, desc="Organizing structures"):
                try:
                    pdb = row['pdb'].lower()
                    h_chain = row['Hchain']
                    l_chain = row['Lchain']
                    is_scfv = row['scfv']
                    heavy_species = row.get('heavy_species', '')
                    light_species = row.get('light_species', '')

                    # Skip scFvs
                    if is_scfv == 'True':
                        skipped += 1
                        continue

                    # Check if structure exists
                    pdb_file = structures_dir / f"{pdb}.pdb"
                    if not pdb_file.exists():
                        skipped += 1
                        state.add_failed_file('organize', str(pdb_file), "File not found")
                        continue

                    # Process heavy chain
                    if h_chain and h_chain != 'NA':
                        try:
                            species = normalize_species(heavy_species)
                            output_file = organized_dir / species / 'heavy' / f"{pdb}_{h_chain}.pdb"

                            if extract_chain_from_pdb(pdb_file, h_chain, output_file):
                                stats[species]['heavy'] += 1
                                processed += 1
                        except Exception as e:
                            state.add_failed_file('organize', f"{pdb}_{h_chain}", str(e))

                    # Process light chain
                    if l_chain and l_chain != 'NA':
                        try:
                            species = normalize_species(light_species)
                            output_file = organized_dir / species / 'light' / f"{pdb}_{l_chain}.pdb"

                            if extract_chain_from_pdb(pdb_file, l_chain, output_file):
                                stats[species]['light'] += 1
                                processed += 1
                        except Exception as e:
                            state.add_failed_file('organize', f"{pdb}_{l_chain}", str(e))

                    # Update state periodically
                    if processed % 1000 == 0:
                        state.state['processed_chains'] = processed
                        state.save()

                except Exception as e:
                    LOGGER.debug(f"Error processing row: {e}")
                    skipped += 1
                    continue

    except Exception as e:
        LOGGER.error(f"Failed to organize antibodies: {e}")
        state.save()
        raise

    # Print statistics
    LOGGER.info(f"✓ Processed: {processed} chains")
    LOGGER.info(f"  Skipped: {skipped} entries")
    for species in ['human', 'mouse', 'llama_alpaca', 'other']:
        if species in stats:
            LOGGER.info(f"  {species}: H={stats[species]['heavy']}, L={stats[species]['light']}")

    state.state['processed_chains'] = processed
    state.mark_step_complete('organize_antibodies')
    return organized_dir


# ============================================================================
# STEP 4: Classify CDR-H3 Conformations
# ============================================================================

def get_ca_coordinates(pdb_file: Path, residue_numbers: List[int]) -> Optional[Dict[int, np.ndarray]]:
    """Extract Cα coordinates for specific residue numbers."""
    coords = {}

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    try:
                        res_num = int(line[22:26].strip())
                        if res_num in residue_numbers:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coords[res_num] = np.array([x, y, z])
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        return None

    # Check if we have all required residues
    for res_num in residue_numbers:
        if res_num not in coords:
            return None

    return coords


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate bond angle in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray, p4: np.ndarray) -> float:
    """Calculate dihedral angle in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(y, x))


def classify_cdrh3(tau: float, alpha: float) -> str:
    """Classify CDR-H3 as kinked or extended."""
    tau_center, alpha_center = 101.0, 39.0
    tau_threshold, alpha_threshold = 20.0, 30.0

    distance = ((tau - tau_center) / tau_threshold)**2 + \
               ((alpha - alpha_center) / alpha_threshold)**2

    return 'kinked' if distance <= 1.0 else 'extended'


def analyze_heavy_chain(pdb_file: Path) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Analyze and classify a heavy chain CDR-H3."""
    required_residues = [115, 116, 117, 118]

    coords = get_ca_coordinates(pdb_file, required_residues)
    if coords is None:
        return None, None, None

    try:
        tau = calculate_angle(coords[115], coords[116], coords[117])
        alpha = calculate_dihedral(coords[115], coords[116], coords[117], coords[118])
        classification = classify_cdrh3(tau, alpha)
        return classification, tau, alpha
    except Exception:
        return None, None, None


def classify_cdrh3_loops(organized_dir: Path, state: PipelineState) -> Path:
    """
    Classify CDR-H3 conformations as kinked or extended.

    Expected time: 5-15 minutes
    CPU: Single-threaded
    """
    LOGGER.info("="*70)
    LOGGER.info("STEP 4: Classifying CDR-H3 Conformations")
    LOGGER.info("="*70)

    if state.is_step_complete('classify_cdrh3'):
        LOGGER.info("CDR-H3 classification already complete")
        return organized_dir

    stats = defaultdict(lambda: defaultdict(int))
    classifications = []

    for species in ['human', 'mouse', 'llama_alpaca', 'other']:
        heavy_dir = organized_dir / species / 'heavy'

        if not heavy_dir.exists():
            continue

        # Check if already classified (has subdirectories)
        if (heavy_dir / 'kinked').exists() or (heavy_dir / 'extended').exists():
            LOGGER.info(f"{species} heavy chains already classified")
            continue

        pdb_files = list(heavy_dir.glob('*.pdb'))
        if not pdb_files:
            continue

        LOGGER.info(f"Classifying {species} heavy chains: {len(pdb_files)} files")

        for pdb_file in tqdm(pdb_files, desc=f"{species} heavy"):
            try:
                classification, tau, alpha = analyze_heavy_chain(pdb_file)

                if classification is None:
                    stats[species]['unclassified'] += 1
                    state.add_failed_file('classify', str(pdb_file), "Missing residues")
                else:
                    classifications.append({
                        'file': pdb_file,
                        'species': species,
                        'classification': classification,
                        'tau': tau,
                        'alpha': alpha
                    })
                    stats[species][classification] += 1

            except Exception as e:
                stats[species]['failed'] += 1
                state.add_failed_file('classify', str(pdb_file), str(e))

    # Reorganize files into kinked/extended subdirectories
    LOGGER.info("Reorganizing files by classification...")

    for item in tqdm(classifications, desc="Moving files"):
        try:
            old_path = item['file']
            species = item['species']
            classification = item['classification']

            new_dir = organized_dir / species / 'heavy' / classification
            new_dir.mkdir(parents=True, exist_ok=True)

            new_path = new_dir / old_path.name
            shutil.move(str(old_path), str(new_path))

        except Exception as e:
            LOGGER.warning(f"Failed to move {old_path}: {e}")

    # Move unclassified to separate directory
    for species in ['human', 'mouse', 'llama_alpaca', 'other']:
        heavy_dir = organized_dir / species / 'heavy'
        if not heavy_dir.exists():
            continue

        remaining_pdbs = list(heavy_dir.glob('*.pdb'))
        if remaining_pdbs:
            unclassified_dir = heavy_dir / 'unclassified'
            unclassified_dir.mkdir(exist_ok=True)
            for pdb in remaining_pdbs:
                shutil.move(str(pdb), str(unclassified_dir / pdb.name))

    # Organize light chains
    LOGGER.info("Organizing light chains...")
    for species in ['human', 'mouse', 'llama_alpaca', 'other']:
        light_dir = organized_dir / species / 'light'

        if not light_dir.exists():
            continue

        # Skip if already organized
        if (light_dir / 'all').exists():
            continue

        light_all_dir = light_dir / 'all'
        light_all_dir.mkdir(parents=True, exist_ok=True)

        pdb_files = list(light_dir.glob('*.pdb'))
        for pdb_file in pdb_files:
            try:
                shutil.move(str(pdb_file), str(light_all_dir / pdb_file.name))
            except Exception as e:
                LOGGER.warning(f"Failed to move {pdb_file}: {e}")

    # Print statistics
    total_kinked = sum(stats[sp]['kinked'] for sp in stats)
    total_extended = sum(stats[sp]['extended'] for sp in stats)
    total_failed = sum(stats[sp]['unclassified'] + stats[sp]['failed'] for sp in stats)

    LOGGER.info(f"✓ Classification complete:")
    LOGGER.info(f"  Kinked: {total_kinked}")
    LOGGER.info(f"  Extended: {total_extended}")
    LOGGER.info(f"  Unclassified: {total_failed}")

    state.mark_step_complete('classify_cdrh3')
    return organized_dir


# ============================================================================
# STEP 5: Cluster Sequences
# ============================================================================

def extract_sequence_from_pdb(pdb_file: Path) -> Optional[str]:
    """Extract amino acid sequence from PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("ab", pdb_file)

        for model in structure:
            for chain in model:
                residues = []
                for residue in chain:
                    if residue.get_id()[0] == " ":  # Standard residue
                        try:
                            residues.append(seq1(residue.get_resname()))
                        except:
                            pass

                if residues:
                    return "".join(residues)
        return None
    except Exception as e:
        return None


def extract_sequences_for_category(category_dir: Path) -> Dict[str, str]:
    """Extract sequences for all PDBs in a category."""
    sequences = {}

    for pdb_file in category_dir.glob('*.pdb'):
        try:
            seq = extract_sequence_from_pdb(pdb_file)
            if seq:
                sequences[pdb_file.stem] = seq
        except Exception as e:
            LOGGER.debug(f"Failed to extract sequence from {pdb_file}: {e}")

    return sequences


def cluster_with_mmseqs(fasta_file: Path, output_prefix: Path,
                        min_seq_id: float = 0.7) -> Optional[Path]:
    """Cluster sequences using MMSeqs2."""
    try:
        # Create database
        db_file = f"{output_prefix}_DB"
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_file), db_file],
            check=True, capture_output=True
        )

        # Cluster
        cluster_db = f"{output_prefix}_cluster"
        tmp_dir = f"{output_prefix}_tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        subprocess.run(
            ["mmseqs", "cluster", db_file, cluster_db, tmp_dir,
             "--min-seq-id", str(min_seq_id),
             "-c", "0.8", "--cov-mode", "1"],
            check=True, capture_output=True
        )

        # Convert to TSV
        tsv_file = Path(f"{output_prefix}_cluster.tsv")
        subprocess.run(
            ["mmseqs", "createtsv", db_file, db_file, cluster_db, str(tsv_file)],
            check=True, capture_output=True
        )

        # Cleanup temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return tsv_file

    except subprocess.CalledProcessError as e:
        LOGGER.error(f"MMSeqs2 failed: {e}")
        return None
    except Exception as e:
        LOGGER.error(f"Clustering error: {e}")
        return None


def parse_clusters(cluster_tsv: Path) -> Dict[str, List[str]]:
    """Parse MMSeqs2 cluster output."""
    clusters = defaultdict(list)

    with open(cluster_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                representative = parts[0]
                member = parts[1]
                clusters[representative].append(member)

    return clusters


def get_resolution_data(summary_file: Path) -> Dict[str, float]:
    """Extract resolution data from summary file."""
    resolution_data = {}

    try:
        df = pd.read_csv(summary_file, sep='\t')

        for _, row in df.iterrows():
            pdb_id = row['pdb']
            h_chain = row['Hchain']
            l_chain = row['Lchain']
            resolution = row.get('resolution', float('inf'))

            # Convert to float
            try:
                if pd.notna(resolution):
                    resolution = float(resolution)
                else:
                    resolution = float('inf')
            except:
                resolution = float('inf')

            # Store for both chains
            if pd.notna(h_chain) and h_chain != 'NA':
                resolution_data[f"{pdb_id}_{h_chain}"] = resolution
            if pd.notna(l_chain) and l_chain != 'NA':
                resolution_data[f"{pdb_id}_{l_chain}"] = resolution

    except Exception as e:
        LOGGER.warning(f"Could not load resolution data: {e}")

    return resolution_data


def select_best_representatives(clusters: Dict, category_dir: Path,
                                 resolution_data: Dict,
                                 species: str, chain_type: str,
                                 classification: str) -> List[Dict]:
    """Select best representative from each cluster."""
    representatives = []

    for rep, members in clusters.items():
        all_members = list(set([rep] + members))

        # Find best resolution
        best_member = None
        best_resolution = float('inf')

        for member in all_members:
            res = resolution_data.get(member, float('inf'))
            if res < best_resolution:
                best_resolution = res
                best_member = member

        if best_member is None:
            best_member = rep
            best_resolution = 'NA'

        # Get file path
        pdb_file = category_dir / f"{best_member}.pdb"

        representatives.append({
            'cluster_representative': rep,
            'best_resolution_member': best_member,
            'resolution': best_resolution if best_resolution != float('inf') else 'NA',
            'species': species,
            'chain_type': chain_type,
            'classification': classification,
            'file': str(pdb_file),
            'cluster_size': len(all_members),
            'members': ';'.join(all_members)
        })

    return representatives


def cluster_sequences(organized_dir: Path, summary_file: Path,
                      state: PipelineState, min_seq_id: float = 0.7,
                      categories_filter: Optional[Set[str]] = None) -> Path:
    """
    Cluster sequences within each category.

    Expected time: 30-60 minutes
    Memory: 4-8 GB
    CPU: Multi-threaded (MMSeqs2)
    """
    LOGGER.info("="*70)
    LOGGER.info("STEP 5: Clustering Sequences at 70% Identity")
    LOGGER.info("="*70)

    output_csv = organized_dir / 'cluster_representatives.csv'

    if state.is_step_complete('cluster_sequences') and output_csv.exists():
        LOGGER.info(f"Clustering already complete: {output_csv}")
        return output_csv

    # Get resolution data
    resolution_data = get_resolution_data(summary_file)

    # Define categories
    categories = []
    for species in ['human', 'mouse', 'llama_alpaca', 'other']:
        for classification in ['kinked', 'extended', 'unclassified']:
            categories.append((species, 'heavy', classification))
        categories.append((species, 'light', 'all'))

    # Filter categories if requested
    if categories_filter:
        categories = [c for c in categories
                     if f"{c[0]}_{c[1]}_{c[2]}" in categories_filter]

    all_representatives = []

    for species, chain_type, classification in categories:
        category_name = f"{species}_{chain_type}_{classification}"
        category_dir = organized_dir / species / chain_type / classification

        if not category_dir.exists():
            continue

        pdb_files = list(category_dir.glob('*.pdb'))
        if len(pdb_files) == 0:
            continue

        LOGGER.info(f"Clustering {category_name}: {len(pdb_files)} sequences")

        try:
            # Extract sequences
            sequences = extract_sequences_for_category(category_dir)
            if not sequences:
                LOGGER.warning(f"No sequences extracted for {category_name}")
                continue

            # Create FASTA
            fasta_file = organized_dir / f"{category_name}_sequences.fasta"
            with open(fasta_file, 'w') as f:
                for seq_id, seq in sequences.items():
                    f.write(f">{seq_id}\n{seq}\n")

            # Cluster
            output_prefix = organized_dir / f"{category_name}_clusters"
            cluster_tsv = cluster_with_mmseqs(fasta_file, output_prefix, min_seq_id)

            if cluster_tsv is None:
                LOGGER.error(f"Clustering failed for {category_name}")
                state.add_failed_file('cluster', category_name, "MMSeqs2 failed")
                continue

            # Parse and select representatives
            clusters = parse_clusters(cluster_tsv)
            representatives = select_best_representatives(
                clusters, category_dir, resolution_data,
                species, chain_type, classification
            )

            all_representatives.extend(representatives)
            LOGGER.info(f"  → {len(clusters)} clusters")

        except Exception as e:
            LOGGER.error(f"Failed to cluster {category_name}: {e}")
            state.add_failed_file('cluster', category_name, str(e))
            continue

    # Save results
    try:
        df = pd.DataFrame(all_representatives)
        df.to_csv(output_csv, index=False)

        LOGGER.info(f"✓ Total clusters: {len(all_representatives)}")
        LOGGER.info(f"  Saved to: {output_csv}")

        state.state['clusters_created'] = len(all_representatives)
        state.mark_step_complete('cluster_sequences')

    except Exception as e:
        LOGGER.error(f"Failed to save cluster results: {e}")
        raise

    return output_csv


# ============================================================================
# STEP 6: Compute MPNN Embeddings
# ============================================================================

def load_mpnn_dependencies():
    """Lazy load JAX/Haiku dependencies."""
    global HAIKU_AVAILABLE, JAX_AVAILABLE

    try:
        import haiku as hk
        import jax
        HAIKU_AVAILABLE = True
        JAX_AVAILABLE = True
        return hk, jax
    except ImportError as e:
        LOGGER.error(f"Failed to import JAX/Haiku: {e}")
        LOGGER.error("Install with: pip install jax haiku dm-haiku")
        return None, None


def load_sabr_ops():
    """Load SAbR ops module."""
    try:
        # Add SAbR to path
        sabr_path = Path(__file__).parent.parent / 'src'
        if str(sabr_path) not in sys.path:
            sys.path.insert(0, str(sabr_path))

        from sabr import ops
        from sabr.types import MPNNEmbeddings
        return ops, MPNNEmbeddings
    except ImportError as e:
        LOGGER.error(f"Failed to import SAbR: {e}")
        return None, None


def get_chain_from_filename(filename: str) -> str:
    """Extract chain ID from filename like '7bz5_H.pdb'."""
    return filename.replace('.pdb', '').split('_')[-1]


def load_existing_embeddings(output_file: Path):
    """Load previously computed embeddings."""
    if not output_file.exists():
        return {}

    embeddings_dict = {}
    try:
        data = np.load(output_file, allow_pickle=True)
        struct_ids = set()

        for key in data.files:
            if key.endswith('_embeddings'):
                struct_id = key.replace('_embeddings', '')
                struct_ids.add(struct_id)

        # Reconstruct embeddings
        _, MPNNEmbeddings = load_sabr_ops()
        if MPNNEmbeddings is None:
            return {}

        for struct_id in struct_ids:
            embeddings = data[f'{struct_id}_embeddings']
            pdb_indices = data[f'{struct_id}_pdb_indices']
            embeddings_dict[struct_id] = MPNNEmbeddings(
                name=struct_id,
                embeddings=embeddings,
                idxs=pdb_indices.tolist(),
                stdev=np.ones_like(embeddings)
            )

        LOGGER.info(f"Loaded {len(embeddings_dict)} existing embeddings")
    except Exception as e:
        LOGGER.warning(f"Could not load existing embeddings: {e}")
        return {}

    return embeddings_dict


def save_embeddings_npz(embeddings_dict: Dict, output_file: Path):
    """Save embeddings to NPZ file."""
    save_dict = {}

    for struct_id, embeddings in embeddings_dict.items():
        emb_array = np.asarray(embeddings.embeddings)
        save_dict[f"{struct_id}_embeddings"] = emb_array
        save_dict[f"{struct_id}_pdb_indices"] = np.array(embeddings.idxs, dtype=object)

    np.savez_compressed(output_file, **save_dict)


def compute_embeddings_for_category(category_name: str, pdb_files: List[str],
                                     output_dir: Path, transformed_embed_fn,
                                     model_params: Dict, key, state: PipelineState,
                                     save_interval: int = 10):
    """Compute MPNN embeddings for a category."""
    output_file = output_dir / f"{category_name}_embeddings.npz"

    # Load existing embeddings
    embeddings_dict = load_existing_embeddings(output_file)
    failed = []

    LOGGER.info(f"Processing {category_name}: {len(pdb_files)} structures")
    if embeddings_dict:
        LOGGER.info(f"  Resuming from {len(embeddings_dict)} already computed")

    structures_since_save = 0

    for pdb_path in tqdm(pdb_files, desc=f"Computing {category_name}"):
        try:
            pdb_path = Path(pdb_path)
            filename = pdb_path.name
            struct_id = filename.replace('.pdb', '')

            # Skip if already computed
            if struct_id in embeddings_dict:
                continue

            # Verify file exists
            if not pdb_path.exists():
                failed.append(str(pdb_path))
                state.add_failed_file('embeddings', str(pdb_path), "File not found")
                continue

            chain = get_chain_from_filename(filename)

            # Compute embeddings
            embeddings = transformed_embed_fn.apply(
                model_params, key, str(pdb_path), chain
            )
            embeddings_dict[struct_id] = embeddings
            structures_since_save += 1

            # Save incrementally
            if structures_since_save >= save_interval:
                save_embeddings_npz(embeddings_dict, output_file)
                structures_since_save = 0
                state.state['embeddings_computed'] = len(embeddings_dict)
                state.save()

        except Exception as e:
            LOGGER.warning(f"Failed to process {pdb_path}: {e}")
            failed.append(str(pdb_path))
            state.add_failed_file('embeddings', str(pdb_path), str(e))
            continue

    # Final save
    save_embeddings_npz(embeddings_dict, output_file)
    LOGGER.info(f"✓ Saved {len(embeddings_dict)} embeddings to {output_file}")

    if failed:
        LOGGER.warning(f"  {len(failed)} structures failed")

    return embeddings_dict


def compute_mpnn_embeddings(cluster_csv: Path, output_dir: Path,
                            state: PipelineState, save_interval: int = 10,
                            categories_filter: Optional[Set[str]] = None):
    """
    Compute MPNN embeddings for cluster representatives.

    Expected time: 30 min - 48 hours (GPU: 30-120 min, CPU: 10-48 hours)
    Memory: 4-16 GB
    GPU: Highly recommended (4-8 GB VRAM)
    Storage: ~3-7 GB output
    """
    LOGGER.info("="*70)
    LOGGER.info("STEP 6: Computing MPNN Embeddings")
    LOGGER.info("="*70)

    embeddings_dir = output_dir / 'mpnn_embeddings'
    embeddings_dir.mkdir(exist_ok=True)

    if state.is_step_complete('compute_embeddings'):
        LOGGER.info("MPNN embeddings already computed")
        return embeddings_dir

    # Load dependencies
    hk, jax = load_mpnn_dependencies()
    if hk is None or jax is None:
        LOGGER.error("Cannot proceed without JAX/Haiku")
        return embeddings_dir

    ops, MPNNEmbeddings = load_sabr_ops()
    if ops is None:
        LOGGER.error("Cannot proceed without SAbR")
        return embeddings_dir

    # Load cluster representatives
    try:
        df = pd.read_csv(cluster_csv)
        LOGGER.info(f"Loaded {len(df)} cluster representatives")
    except Exception as e:
        LOGGER.error(f"Failed to load cluster CSV: {e}")
        return embeddings_dir

    # Initialize model
    LOGGER.info("Initializing MPNN model (this may take a few minutes)...")
    try:
        transformed_embed_fn = hk.transform(ops.embed_fn)
        key = jax.random.PRNGKey(0)

        # Initialize with first sample
        sample_row = df.iloc[0]
        sample_pdb = sample_row['file']
        sample_chain = get_chain_from_filename(Path(sample_pdb).name)

        LOGGER.info(f"Initializing with sample: {sample_pdb}")
        model_params = transformed_embed_fn.init(key, sample_pdb, sample_chain)
        LOGGER.info("✓ Model initialized")

    except Exception as e:
        LOGGER.error(f"Failed to initialize model: {e}")
        return embeddings_dir

    # Define categories
    categories = []
    for species in ['human', 'mouse', 'llama_alpaca', 'other']:
        for classification in ['kinked', 'extended', 'unclassified']:
            categories.append((species, 'heavy', classification))
        categories.append((species, 'light', 'all'))

    # Filter if requested
    if categories_filter:
        categories = [c for c in categories
                     if f"{c[0]}_{c[1]}_{c[2]}" in categories_filter]

    # Process each category
    for species, chain_type, classification in categories:
        category_name = f"{species}_{chain_type}_{classification}"

        # Filter dataframe
        mask = (
            (df['species'] == species) &
            (df['chain_type'] == chain_type) &
            (df['classification'] == classification)
        )
        category_df = df[mask]

        if len(category_df) == 0:
            continue

        pdb_files = category_df['file'].tolist()

        try:
            compute_embeddings_for_category(
                category_name=category_name,
                pdb_files=pdb_files,
                output_dir=embeddings_dir,
                transformed_embed_fn=transformed_embed_fn,
                model_params=model_params,
                key=key,
                state=state,
                save_interval=save_interval
            )
            state.state['last_category'] = category_name
            state.save()

        except Exception as e:
            LOGGER.error(f"Failed to process category {category_name}: {e}")
            state.add_failed_file('embeddings', category_name, str(e))
            continue

    LOGGER.info("✓ All embeddings computed")
    state.mark_step_complete('compute_embeddings')
    return embeddings_dir


# ============================================================================
# Main Pipeline
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process SAbDab antibody structures and compute MPNN embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Base output directory for all results'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading structures (use existing)'
    )

    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering step'
    )

    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip MPNN embedding computation'
    )

    parser.add_argument(
        '--min-seq-id',
        type=float,
        default=0.7,
        help='Minimum sequence identity for clustering (default: 0.7)'
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='Save embeddings every N structures (default: 10)'
    )

    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of categories to process (e.g., human_heavy_kinked,mouse_light_all)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run (automatic if state file exists)'
    )

    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_args()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize state
    state_file = output_dir / '.pipeline_state.json'
    state = PipelineState(state_file)

    LOGGER.info("="*70)
    LOGGER.info("SAbDab Antibody Processing Pipeline")
    LOGGER.info("="*70)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"State file: {state_file}")

    # Parse categories filter
    categories_filter = None
    if args.categories:
        categories_filter = set(args.categories.split(','))
        LOGGER.info(f"Processing only categories: {categories_filter}")

    try:
        # Step 1: Download summary
        summary_file = download_summary(output_dir, state)

        # Step 2: Download structures
        if args.skip_download:
            LOGGER.info("Skipping structure download (--skip-download)")
            structures_dir = output_dir / 'all_structures' / 'imgt'
        else:
            structures_dir = download_structures(summary_file, output_dir, state)

        # Step 3: Organize antibodies
        organized_dir = organize_antibodies(summary_file, structures_dir, output_dir, state)

        # Step 4: Classify CDR-H3
        organized_dir = classify_cdrh3_loops(organized_dir, state)

        # Step 5: Cluster sequences
        if args.skip_clustering:
            LOGGER.info("Skipping clustering (--skip-clustering)")
            cluster_csv = organized_dir / 'cluster_representatives.csv'
        else:
            cluster_csv = cluster_sequences(
                organized_dir, summary_file, state,
                min_seq_id=args.min_seq_id,
                categories_filter=categories_filter
            )

        # Step 6: Compute embeddings
        if args.skip_embeddings:
            LOGGER.info("Skipping embeddings (--skip-embeddings)")
        else:
            embeddings_dir = compute_mpnn_embeddings(
                cluster_csv, output_dir, state,
                save_interval=args.save_interval,
                categories_filter=categories_filter
            )

        # Final summary
        LOGGER.info("="*70)
        LOGGER.info("Pipeline Complete!")
        LOGGER.info("="*70)
        LOGGER.info(f"Results in: {output_dir}")

        # Print failed files summary
        if state.state['failed_files']:
            LOGGER.info("\nFailed files by step:")
            for step, failures in state.state['failed_files'].items():
                LOGGER.info(f"  {step}: {len(failures)} failures")

        LOGGER.info("\nOutput structure:")
        LOGGER.info(f"  {output_dir}/")
        LOGGER.info(f"    summary.tsv")
        LOGGER.info(f"    all_structures/imgt/  (~8.2 GB)")
        LOGGER.info(f"    organized_antibodies/  (~2.4 GB)")
        LOGGER.info(f"      cluster_representatives.csv")
        LOGGER.info(f"    mpnn_embeddings/  (~3-7 GB)")

    except KeyboardInterrupt:
        LOGGER.warning("\nPipeline interrupted by user")
        state.save()
        LOGGER.info("Progress saved. Resume by running the same command.")
        sys.exit(1)

    except Exception as e:
        LOGGER.error(f"\nPipeline failed: {e}")
        state.save()
        raise


if __name__ == '__main__':
    main()
