#!/usr/bin/env python3
"""Command-line interface for SAbR antibody renumbering."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from sabr.errors import SAbRError
from sabr.options import RenumberOptions
from sabr.renumber import renumber_file
from sabr.structure.residues import ResidueRange

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, force=True)


def _validate_chain_id(ctx, _param, value: str) -> str:
    if len(value) != 1:
        ctx.fail("Chain identifier must be exactly one character.")
    return value


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Structure-based Antibody Renumbering (SAbR) renumbers antibody "
        "structure files using the 3D coordinates of backbone atoms."
    ),
)
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="Input structure file (.pdb or .cif).",
)
@click.option(
    "-c",
    "--input-chain",
    "input_chain",
    required=True,
    callback=_validate_chain_id,
    help="Chain identifier to renumber.",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Destination structure file (.pdb or .cif).",
)
@click.option(
    "-n",
    "--numbering-scheme",
    "numbering_scheme",
    default="imgt",
    show_default="IMGT",
    type=click.Choice(
        ["imgt", "chothia", "kabat", "martin", "aho", "wolfguy"],
        case_sensitive=False,
    ),
    help="Numbering scheme.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite the output file.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.option(
    "--residue-range",
    "residue_range",
    nargs=2,
    type=int,
    default=None,
    help=(
        "Original residue-number range to renumber as START END "
        "(inclusive). Omit to process all residues; residues outside the "
        "range are preserved unchanged."
    ),
)
@click.option(
    "--disable-deterministic-renumbering",
    "disable_deterministic_renumbering",
    is_flag=True,
    help="Disable deterministic renumbering corrections.",
)
@click.option(
    "--random-seed",
    "random_seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for JAX operations.",
)
@click.option(
    "-t",
    "--chain-type",
    "chain_type",
    default="auto",
    show_default=True,
    type=click.Choice(
        ["H", "K", "L", "auto"],
        case_sensitive=False,
    ),
    help="Chain type embedding label to use: H, K, L, or auto.",
)
@click.option(
    "--disable-custom-gap-penalties",
    "disable_custom_gap_penalties",
    is_flag=True,
    help=(
        "Disable custom CDR gap penalties. By default, gap-open penalties "
        "are zero only in IMGT CDR regions."
    ),
)
@click.version_option(package_name="sabr-kit", prog_name="sabr")
def main(
    input_path: str,
    input_chain: str,
    output_file: str,
    numbering_scheme: str,
    overwrite: bool,
    verbose: bool,
    residue_range: tuple[int, int] | None,
    disable_deterministic_renumbering: bool,
    random_seed: int,
    chain_type: str,
    disable_custom_gap_penalties: bool,
) -> None:
    """Run the command-line workflow for renumbering antibody structures."""
    _configure_logging(verbose)

    LOGGER.info("Using random seed: %s", random_seed)

    try:
        options = RenumberOptions.from_values(
            numbering_scheme=numbering_scheme,
            chain_type=chain_type,
            deterministic_corrections=not disable_deterministic_renumbering,
            custom_gap_penalties=not disable_custom_gap_penalties,
            residue_range=(
                ResidueRange(*residue_range) if residue_range is not None else None
            ),
            random_seed=random_seed,
            overwrite=overwrite,
        )
        result = renumber_file(
            input_path=Path(input_path),
            chain_id=input_chain,
            output_path=Path(output_file),
            options=options,
        )
    except (SAbRError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    LOGGER.info(
        "Finished renumbering; output=%s chain_type=%s changed_residues=%s",
        result.output_path,
        result.chain_type.value,
        result.changed_residue_count,
    )


if __name__ == "__main__":
    main()
