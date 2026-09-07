"""The SAbR command-line interface."""

import logging
import os
import uuid
from pathlib import Path

import click
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser

from sabr import constants, renumber_structure

LOGGER = logging.getLogger(__name__)


def _read_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return PDBParser(QUIET=True).get_structure(path.stem, path)
    if suffix in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True).get_structure(path.stem, path)
    raise ValueError("Input must use the .pdb, .cif, or .mmcif extension.")


def _validate_pdb_output(structure) -> None:
    if sum(1 for _ in structure.get_atoms()) > 99_999:
        raise ValueError(
            "PDB output supports at most 99,999 atoms; use mmCIF output."
        )
    for chain in structure.get_chains():
        if len(chain.id) != 1 or ord(chain.id) < 32 or ord(chain.id) > 126:
            raise ValueError(
                "PDB output requires one printable ASCII chain character; "
                f"'{chain.id}' requires mmCIF output."
            )
        for residue in chain:
            number = residue.id[1]
            if not -999 <= number <= 9999:
                raise ValueError(
                    f"PDB output cannot represent residue number {number}; "
                    "use mmCIF output."
                )
            insertion_code = residue.id[2].strip()
            if len(insertion_code) > 1 or (
                insertion_code and not 32 <= ord(insertion_code) <= 126
            ):
                raise ValueError(
                    "PDB output requires blank or one printable ASCII "
                    "insertion character; use mmCIF output."
                )


def _has_extended_insertion_codes(structure) -> bool:
    return any(
        len(residue.id[2].strip()) > 1 for residue in structure.get_residues()
    )


def _resolve_output_path(structure, path: Path, no_mmcif: bool = False) -> Path:
    if (
        path.suffix.lower() == ".pdb"
        and not no_mmcif
        and _has_extended_insertion_codes(structure)
    ):
        return path.with_suffix(".cif")
    return path


def _write_structure(structure, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        _validate_pdb_output(structure)
        writer = PDBIO()
    elif suffix in (".cif", ".mmcif"):
        writer = MMCIFIO()
    else:
        raise ValueError("Output must use the .pdb, .cif, or .mmcif extension.")
    writer.set_structure(structure)
    writer.save(str(path))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="sabr-kit")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input PDB or mmCIF file.",
)
@click.option(
    "-c", "--chain", required=True, help="Chain identifier to renumber."
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output PDB or mmCIF file.",
)
@click.option(
    "-n",
    "--scheme",
    default="imgt",
    show_default=True,
    type=click.Choice(constants.NUMBERING_SCHEMES, case_sensitive=False),
)
@click.option(
    "-t",
    "--chain-type",
    default="auto",
    show_default=True,
    metavar="TYPES",
    help=(
        "Comma-separated antibody domain representations, such as H,K or "
        "HK,HL; TCR types A, B, G, and D use the K reference."
    ),
)
@click.option(
    "--noise-level",
    default="0.0",
    show_default=True,
    type=click.Choice(
        tuple(f"{level:.1f}" for level in constants.NOISE_LEVELS)
    ),
)
@click.option(
    "-m",
    "--mode",
    default="sabr",
    show_default=True,
    type=click.Choice(constants.MODES, case_sensitive=False),
    help="Select a complete encoder, references, and gap-penalty set.",
)
@click.option("--residue-range", nargs=2, type=int, metavar="START END")
@click.option(
    "--scfv",
    is_flag=True,
    help="Equivalent to --chain-type HK,HL,KH,LH.",
)
@click.option(
    "--dangerously-allow-structural-gaps",
    is_flag=True,
    help="Run despite structural gaps; results may be invalid.",
)
@click.option(
    "--no-mmcif",
    is_flag=True,
    help="Forbid automatic mmCIF output for extended insertion codes.",
)
@click.option("--overwrite", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def main(
    input_path: Path,
    chain: str,
    output_path: Path,
    scheme: str,
    chain_type: str,
    noise_level: str,
    mode: str,
    residue_range: tuple | None,
    scfv: bool,
    dangerously_allow_structural_gaps: bool,
    no_mmcif: bool,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Renumber one antibody chain in a PDB or mmCIF structure."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        force=True,
    )
    if dangerously_allow_structural_gaps:
        LOGGER.warning(
            "Structural-gap safety check disabled by "
            "--dangerously-allow-structural-gaps; results may be invalid."
        )
    temporary = None
    try:
        structure = _read_structure(input_path)
        if input_path.suffix.lower() in (".cif", ".mmcif"):
            LOGGER.warning(
                "Non-atomic mmCIF categories are not preserved when parsed "
                "with Biopython."
            )
        LOGGER.info("Numerical backend: NumPy/SciPy (CPU)")
        result = renumber_structure(
            structure,
            chain,
            scheme=scheme,
            chain_type=chain_type,
            noise_level=float(noise_level),
            mode=mode,
            residue_range=residue_range,
            scfv=scfv,
            dangerously_allow_structural_gaps=(
                dangerously_allow_structural_gaps
            ),
        )
        resolved_output_path = _resolve_output_path(
            result, output_path, no_mmcif=no_mmcif
        )
        if resolved_output_path.exists() and not overwrite:
            raise click.ClickException(
                f"Output already exists: {resolved_output_path}. "
                "Use --overwrite to replace it."
            )
        if resolved_output_path != output_path:
            LOGGER.warning(
                "PDB cannot represent multi-character insertion codes; "
                "automatically saving mmCIF output to %s. Use --no-mmcif "
                "to forbid this behavior.",
                resolved_output_path,
            )
        temporary = resolved_output_path.with_name(
            f".{resolved_output_path.stem}.{uuid.uuid4().hex}"
            f".tmp{resolved_output_path.suffix}"
        )
        _write_structure(result, temporary)
        os.replace(temporary, resolved_output_path)
    except click.ClickException:
        raise
    except Exception as error:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError as cleanup_error:
                LOGGER.warning(
                    "Could not remove temporary output %s: %s",
                    temporary,
                    cleanup_error,
                )
        if verbose:
            LOGGER.exception("Renumbering failed")
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
