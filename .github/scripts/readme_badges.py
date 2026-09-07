"""Generate Shields badge data from coverage and registry metadata."""

import argparse
import json
import math
from pathlib import Path


def badge(label, message, color):
    return {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
    }


def coverage_badge(report):
    percent = float(report["totals"]["percent_covered"])
    if not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("Coverage must be a finite percentage from 0 to 100")
    return badge(
        "coverage", f"{percent:.1f}%", "brightgreen" if percent >= 95 else "red"
    )


def docker_badge(image, tag, revision):
    if not tag or not revision:
        return badge("docker", "no version tag", "lightgrey")
    # Buildx returns one config or a map of platform names to configs.
    configs = [image] if "config" in image else list(image.values())
    labels = [config.get("config", {}).get("Labels", {}) for config in configs]
    version_key = "org.opencontainers.image.version"
    revision_key = "org.opencontainers.image.revision"
    if not labels or any(
        not label or not label.get(version_key) or not label.get(revision_key)
        for label in labels
    ):
        return badge("docker", "unknown", "lightgrey")
    current = all(
        label[version_key].removeprefix("v") == tag.removeprefix("v")
        and label[revision_key] == revision
        for label in labels
    )
    status = "up to date" if current else "out of date"
    return badge(
        "docker", f"{status} ({tag})", "brightgreen" if current else "orange"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    coverage = commands.add_parser("coverage")
    coverage.add_argument("report", type=Path)
    coverage.add_argument("output", type=Path)
    docker = commands.add_parser("docker")
    docker.add_argument("image", type=Path)
    docker.add_argument("tag")
    docker.add_argument("revision")
    docker.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "coverage":
        result = coverage_badge(json.loads(args.report.read_text()))
    else:
        result = docker_badge(
            json.loads(args.image.read_text()), args.tag, args.revision
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
