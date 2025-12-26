#!/usr/bin/env python3
"""Dismiss AI code review blocks on pull requests."""

import os
import sys

from github import Auth, Github


def main() -> None:
    """Dismiss any blocking AI reviews on the PR."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)

    repo_name = os.environ.get("REPO_NAME")
    pr_number_str = os.environ.get("PR_NUMBER")

    if not repo_name or not pr_number_str:
        print("Error: REPO_NAME and PR_NUMBER must be set")
        sys.exit(1)

    try:
        pr_number = int(pr_number_str)
    except ValueError:
        print(f"Error: PR_NUMBER '{pr_number_str}' is not a valid integer")
        sys.exit(1)

    print(f"Dismissing AI reviews for {repo_name}#{pr_number}")

    try:
        gh = Github(auth=Auth.Token(token))
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)

        dismissed_count = 0
        reviews = pr.get_reviews()

        for review in reviews:
            if (
                review.state == "CHANGES_REQUESTED"
                and review.body
                and "AI Code Review" in review.body
            ):
                try:
                    review.dismiss(
                        "Dismissed by maintainer via /dismiss command"
                    )
                    print(f"Dismissed review {review.id}")
                    dismissed_count += 1
                except Exception as e:
                    print(f"Could not dismiss review {review.id}: {e}")

        if dismissed_count > 0:
            pr.create_issue_comment(
                f"## :white_check_mark: AI Review Dismissed\n\n"
                f"Dismissed {dismissed_count} blocking AI review(s). "
                f"Merging is now unblocked."
            )
            print(f"Dismissed {dismissed_count} AI review(s)")
        else:
            pr.create_issue_comment(
                "## :information_source: No Blocking Reviews\n\n"
                "No blocking AI reviews found to dismiss."
            )
            print("No blocking AI reviews found")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
