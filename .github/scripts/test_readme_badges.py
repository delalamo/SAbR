"""Check badge accuracy and publication against an isolated local Git remote."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from readme_badges import coverage_badge, docker_badge

PUBLISH_SCRIPT = Path(__file__).with_name("publish-badges.sh").resolve()


def image(version="0.4.6", revision="abc123"):
    return {
        "config": {
            "Labels": {
                "org.opencontainers.image.version": version,
                "org.opencontainers.image.revision": revision,
            }
        }
    }


class BadgeTests(unittest.TestCase):
    def test_coverage_uses_measured_percentage(self):
        result = coverage_badge({"totals": {"percent_covered": 97.36}})
        self.assertEqual(result["message"], "97.4%")
        self.assertEqual(result["color"], "brightgreen")
        result = coverage_badge({"totals": {"percent_covered": 94}})
        self.assertEqual(result["color"], "red")

    def test_invalid_coverage_is_rejected(self):
        for value in [-1, 101, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                coverage_badge({"totals": {"percent_covered": value}})

    def test_matching_image(self):
        result = docker_badge(image(), "v0.4.6", "abc123")
        self.assertEqual(result["message"], "up to date (v0.4.6)")
        self.assertEqual(result["color"], "brightgreen")

    def test_old_version_or_retagged_commit_is_outdated(self):
        for config in [image(version="0.4.5"), image(revision="old")]:
            result = docker_badge(config, "v0.4.6", "abc123")
            self.assertEqual(result["message"], "out of date (v0.4.6)")
            self.assertEqual(result["color"], "orange")

    def test_missing_metadata_is_unknown(self):
        for config in [{}, {"config": {}}, image(revision="")]:
            result = docker_badge(config, "v0.4.6", "abc123")
            self.assertEqual(result["message"], "unknown")

    def test_no_tags(self):
        self.assertEqual(
            docker_badge(image(), "", "")["message"], "no version tag"
        )

    def test_all_platforms_must_match(self):
        configs = {"linux/amd64": image(), "linux/arm64": image()}
        self.assertEqual(
            docker_badge(configs, "v0.4.6", "abc123")["color"], "brightgreen"
        )
        configs["linux/arm64"] = image(version="0.4.5")
        self.assertEqual(
            docker_badge(configs, "v0.4.6", "abc123")["color"], "orange"
        )

    def test_publication_preserves_other_badge_and_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote.git"
            repo = root / "checkout"
            output = root / "output"
            repo.mkdir()
            output.mkdir()

            def run(*args):
                return subprocess.run(
                    args, cwd=repo, check=True, capture_output=True, text=True
                ).stdout.strip()

            run("git", "init", "--bare", str(remote))
            run("git", "init")
            run("git", "config", "user.name", "Badge test")
            run("git", "config", "user.email", "test@example.com")
            run("git", "remote", "add", "origin", str(remote))
            (repo / "source.txt").write_text("original\n")
            run("git", "add", "source.txt")
            run("git", "commit", "-m", "Initial source")
            head = run("git", "rev-parse", "HEAD")
            (repo / "source.txt").write_text("staged edit\n")
            run("git", "add", "source.txt")
            staged = run("git", "diff", "--cached")

            coverage = {"schemaVersion": 1, "message": "97.4%"}
            (output / "coverage.json").write_text(json.dumps(coverage))
            run("bash", str(PUBLISH_SCRIPT), str(output))
            (output / "coverage.json").unlink()
            (output / "docker.json").write_text('{"message":"up to date"}')
            run("bash", str(PUBLISH_SCRIPT), str(output))
            published = run("git", "ls-remote", "origin", "refs/heads/badges")
            run("bash", str(PUBLISH_SCRIPT), str(output))
            self.assertEqual(
                published,
                run("git", "ls-remote", "origin", "refs/heads/badges"),
            )
            self.assertEqual(
                json.loads(run("git", "show", "FETCH_HEAD:coverage.json")),
                coverage,
            )
            self.assertEqual(
                run("git", "ls-tree", "--name-only", "FETCH_HEAD"),
                "coverage.json\ndocker.json",
            )
            self.assertEqual(run("git", "rev-parse", "HEAD"), head)
            self.assertEqual(run("git", "diff", "--cached"), staged)
            self.assertEqual((repo / "source.txt").read_text(), "staged edit\n")


if __name__ == "__main__":
    unittest.main()
