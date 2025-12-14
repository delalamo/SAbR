#!/usr/bin/env python3
"""Configuration dataclasses for SAbR pipeline.

This module provides configuration dataclasses that consolidate
pipeline parameters, making it easier to manage and pass configuration
throughout the application.
"""

from dataclasses import dataclass, field
from typing import Optional

from sabr import constants


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for the SoftAlign alignment process.

    Attributes:
        params_name: Name of the SoftAlign model parameters file.
        params_path: Package path containing the parameters.
        embeddings_name: Name of the reference embeddings file.
        embeddings_path: Package path containing the embeddings.
        temperature: Alignment temperature (lower = stricter matching).
        random_seed: Random seed for reproducibility.
    """

    params_name: str = "CONT_SW_05_T_3_1"
    params_path: str = "softalign.models"
    embeddings_name: str = "embeddings.npz"
    embeddings_path: str = "sabr.assets"
    temperature: float = 1e-4
    random_seed: int = 0


@dataclass(frozen=True)
class NumberingConfig:
    """Configuration for antibody numbering.

    Attributes:
        scheme: Numbering scheme (imgt, chothia, kabat, etc.).
        chain_type: Chain type filter (heavy, light, auto).
        deterministic_loop_renumbering: Apply deterministic corrections.
        extended_insertions: Use extended insertion codes for long CDRs.
    """

    scheme: str = "imgt"
    chain_type: constants.ChainType = constants.ChainType.AUTO
    deterministic_loop_renumbering: bool = True
    extended_insertions: bool = False


@dataclass(frozen=True)
class IOConfig:
    """Configuration for input/output operations.

    Attributes:
        input_pdb: Path to input PDB file.
        input_chain: Chain identifier to process.
        output_file: Path to output file.
        overwrite: Whether to overwrite existing output.
        max_residues: Maximum residues to process (0 = all).
    """

    input_pdb: str
    input_chain: str
    output_file: str
    overwrite: bool = False
    max_residues: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    """Complete configuration for the SAbR renumbering pipeline.

    This dataclass consolidates all configuration options needed to run
    the antibody renumbering pipeline, including alignment settings,
    numbering scheme options, and I/O parameters.

    Example:
        config = PipelineConfig(
            io=IOConfig(
                input_pdb="antibody.pdb",
                input_chain="H",
                output_file="renumbered.pdb",
            ),
            numbering=NumberingConfig(scheme="imgt"),
        )
    """

    io: IOConfig
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    numbering: NumberingConfig = field(default_factory=NumberingConfig)
    verbose: bool = False

    @classmethod
    def from_cli_args(
        cls,
        input_pdb: str,
        input_chain: str,
        output_file: str,
        numbering_scheme: str = "imgt",
        chain_type: str = "auto",
        overwrite: bool = False,
        verbose: bool = False,
        max_residues: int = 0,
        extended_insertions: bool = False,
        deterministic_loop_renumbering: bool = True,
    ) -> "PipelineConfig":
        """Create a PipelineConfig from CLI arguments.

        This factory method translates CLI argument values into the
        appropriate configuration objects.
        """
        chain_type_enum = constants.ChainType(chain_type)

        return cls(
            io=IOConfig(
                input_pdb=input_pdb,
                input_chain=input_chain,
                output_file=output_file,
                overwrite=overwrite,
                max_residues=max_residues,
            ),
            numbering=NumberingConfig(
                scheme=numbering_scheme,
                chain_type=chain_type_enum,
                deterministic_loop_renumbering=deterministic_loop_renumbering,
                extended_insertions=extended_insertions,
            ),
            verbose=verbose,
        )
