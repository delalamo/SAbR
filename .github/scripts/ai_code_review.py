#!/usr/bin/env python3
"""AI-powered code review using Anthropic's Claude API."""

import json
import os
import re
import sys
from pathlib import Path

import anthropic
from github import Auth, Github


def get_pr_diff() -> str:
    """Read the PR diff from file."""
    diff_file = Path("pr_diff.txt")
    if not diff_file.exists():
        print("Error: pr_diff.txt not found")
        sys.exit(1)
    return diff_file.read_text()


def parse_diff_files(diff: str) -> dict[str, list[int]]:
    """Parse diff to extract changed files and their line numbers."""
    files = {}
    current_file = None
    current_line = 0

    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:]
            files[current_file] = []
        elif line.startswith("@@") and current_file:
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1))
        elif (
            current_file and line.startswith("+") and not line.startswith("+++")
        ):
            files[current_file].append(current_line)
            current_line += 1
        elif current_file and not line.startswith("-"):
            current_line += 1

    return files


def analyze_code_with_claude(
    diff: str, changed_files: dict[str, list[int]], client: anthropic.Anthropic
) -> dict:
    """Send the diff to Claude for analysis and get structured feedback."""
    files_context = "\n".join(
        [
            f"- {f}: lines {min(lines) if lines else 0}-"
            f"{max(lines) if lines else 0}"
            for f, lines in changed_files.items()
            if lines
        ]
    )

    system_prompt = (
        "You are a code reviewer. Respond ONLY with JSON.\n\n"
        "OUTPUT FORMAT:\n"
        "{\n"
        '  "summary": "<MAX 15 WORDS>",\n'
        '  "approval": "APPROVE" or "REQUEST_CHANGES",\n'
        '  "comments": [\n'
        "    {\n"
        '      "file": "path/to/file.py",\n'
        '      "line": 42,\n'
        '      "body": "Issue description",\n'
        '      "severity": "trivial" or "substantial",\n'
        '      "old_code": "original code (only if trivial)",\n'
        '      "new_code": "fixed code (only if trivial)"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "SEVERITY RULES:\n"
        '- "trivial": typos, simple bugs, missing imports, obvious fixes\n'
        "  -> MUST include old_code and new_code for auto-fix\n"
        '- "substantial": design issues, complex bugs, security concerns\n'
        "  -> only include body description, human will fix\n\n"
        "IMPORTANT:\n"
        "- old_code/new_code = exact single-line or multi-line snippet\n"
        "- old_code must match file content EXACTLY (including whitespace)\n"
        "- Do NOT invent issues - empty comments array is fine\n"
        "- Only real bugs, not style preferences\n\n"
        "EXAMPLE:\n"
        "{\n"
        '  "summary": "Has typo and potential security issue.",\n'
        '  "approval": "REQUEST_CHANGES",\n'
        '  "comments": [\n'
        '    {"file": "src/main.py", "line": 10, "body": "Typo in var", '
        '"severity": "trivial", "old_code": "usre", "new_code": "user"},\n'
        '    {"file": "src/auth.py", "line": 55, "body": "SQL injection", '
        '"severity": "substantial"}\n'
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Files changed:\n{files_context}\n\n"
        f"```diff\n{diff}\n```\n\n"
        "Review and respond with JSON. Categorize each issue."
    )

    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )

    response_text = message.content[0].text
    print(f"Raw Claude response:\n{response_text}\n")

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        response_text = json_match.group(1)

    try:
        result = json.loads(response_text.strip())
        trivial = sum(
            1
            for c in result.get("comments", [])
            if c.get("severity") == "trivial"
        )
        substantial = len(result.get("comments", [])) - trivial
        print(f"Parsed: {trivial} trivial, {substantial} substantial issues")
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return {
            "summary": "Unable to parse AI response. Please review manually.",
            "approval": "APPROVE",
            "comments": [],
        }


def apply_trivial_fixes(
    comments: list[dict],
    repo,
    pr,
    changed_files: dict[str, list[int]],
) -> list[dict]:
    """Apply trivial fixes directly to the PR branch.

    Returns list of comments that could not be auto-fixed.
    """
    branch_name = pr.head.ref
    fixes_applied = []
    remaining_comments = []

    for comment in comments:
        severity = comment.get("severity", "substantial")
        if severity != "trivial":
            remaining_comments.append(comment)
            continue

        file_path = comment.get("file", "")
        old_code = comment.get("old_code", "")
        new_code = comment.get("new_code", "")

        if not old_code or not new_code or old_code == new_code:
            print(f"  Skipping {file_path}: missing or identical code")
            remaining_comments.append(comment)
            continue

        if file_path not in changed_files:
            print(f"  Skipping {file_path}: not in changed files")
            remaining_comments.append(comment)
            continue

        try:
            # Get current file content
            file_content = repo.get_contents(file_path, ref=branch_name)
            content = file_content.decoded_content.decode("utf-8")

            # Apply the fix
            if old_code not in content:
                print(f"  Skipping {file_path}: old_code not found in file")
                remaining_comments.append(comment)
                continue

            new_content = content.replace(old_code, new_code, 1)

            if new_content == content:
                print(f"  Skipping {file_path}: no change after replacement")
                remaining_comments.append(comment)
                continue

            fixes_applied.append(
                {
                    "file": file_path,
                    "body": comment.get("body", ""),
                    "old_code": old_code,
                    "new_code": new_code,
                    "sha": file_content.sha,
                    "new_content": new_content,
                }
            )
            print(f"  Prepared fix for {file_path}: {comment.get('body', '')}")

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            remaining_comments.append(comment)

    # Commit all fixes in one commit
    if fixes_applied:
        print(f"\nCommitting {len(fixes_applied)} auto-fixes...")
        for fix in fixes_applied:
            try:
                repo.update_file(
                    path=fix["file"],
                    message=f"Auto-fix: {fix['body']}",
                    content=fix["new_content"],
                    sha=fix["sha"],
                    branch=branch_name,
                )
                print(f"  Committed fix to {fix['file']}")
            except Exception as e:
                print(f"  Failed to commit {fix['file']}: {e}")
                # Add back to remaining comments if commit failed
                remaining_comments.append(
                    {
                        "file": fix["file"],
                        "body": fix["body"],
                        "severity": "trivial",
                        "old_code": fix["old_code"],
                        "new_code": fix["new_code"],
                    }
                )

    return remaining_comments


def dismiss_previous_ai_reviews(pr) -> None:
    """Dismiss any previous AI reviews that requested changes."""
    try:
        reviews = pr.get_reviews()
        for review in reviews:
            # Check if this is an AI review that requested changes
            if (
                review.state == "CHANGES_REQUESTED"
                and review.body
                and "AI Code Review" in review.body
            ):
                try:
                    review.dismiss("Superseded by new AI review")
                    print(f"Dismissed previous AI review {review.id}")
                except Exception as e:
                    print(f"Could not dismiss review {review.id}: {e}")
    except Exception as e:
        print(f"Warning: Could not check previous reviews: {e}")


def post_review(
    review_data: dict,
    repo_name: str,
    pr_number: int,
    changed_files: dict[str, list[int]],
) -> None:
    """Post review with auto-fixes for trivial issues, comments for rest."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)

    gh = Github(auth=Auth.Token(token))
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Dismiss any previous AI reviews that blocked merging
    dismiss_previous_ai_reviews(pr)

    commit = pr.get_commits().reversed[0]

    summary = review_data.get("summary", "No summary provided")
    approval = review_data.get("approval", "APPROVE")
    comments = review_data.get("comments", [])

    # Separate trivial and substantial issues
    trivial_comments = [c for c in comments if c.get("severity") == "trivial"]
    substantial_comments = [
        c for c in comments if c.get("severity") != "trivial"
    ]

    # Apply trivial fixes
    auto_fix_summary = ""
    if trivial_comments:
        print(f"\nProcessing {len(trivial_comments)} trivial fixes...")
        remaining = apply_trivial_fixes(
            trivial_comments, repo, pr, changed_files
        )
        fixed_count = len(trivial_comments) - len(remaining)
        if fixed_count > 0:
            auto_fix_summary = (
                f"\n\n:wrench: **Auto-fixed {fixed_count} trivial issue(s)**"
            )
        # Add any unfixed trivial issues to substantial for commenting
        substantial_comments.extend(remaining)

    # Build review body
    if approval == "APPROVE" and not substantial_comments:
        body = f"## :white_check_mark: AI Code Review - Approved\n\n{summary}"
        body += "\n\n*No issues found. The changes look good!*"
    elif substantial_comments:
        body = f"## :warning: AI Code Review - Changes Requested\n\n{summary}"
    else:
        body = f"## :white_check_mark: AI Code Review - Approved\n\n{summary}"

    body += auto_fix_summary
    body += (
        "\n\n---\n*This review was generated by Claude AI. "
        "Please use your judgment when considering these suggestions.*"
    )

    # Prepare review comments for substantial issues
    review_comments = []
    for comment in substantial_comments:
        file_path = comment.get("file", "")
        line = comment.get("line", 0)
        comment_body = comment.get("body", "")

        if file_path not in changed_files:
            continue

        valid_lines = changed_files.get(file_path, [])
        if not valid_lines:
            continue

        if line not in valid_lines:
            line = min(valid_lines, key=lambda x: abs(x - line))

        review_comments.append(
            {
                "path": file_path,
                "line": line,
                "body": comment_body,
            }
        )

    # Determine review event
    # Note: GitHub Actions cannot use APPROVE, so we use COMMENT for approvals
    if substantial_comments and review_comments:
        event = "REQUEST_CHANGES"
    else:
        event = "COMMENT"

    # Post the review
    if review_comments:
        pr.create_review(
            commit=commit,
            body=body,
            event=event,
            comments=review_comments,
        )
        print(f"Posted review with {len(review_comments)} comments")
    else:
        pr.create_review(
            commit=commit,
            body=body,
            event=event,
        )
        print("Posted review summary")

    print(f"Review status: {event}")


def main() -> None:
    """Main entry point for the AI code review."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
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

    print(f"Starting AI code review for {repo_name}#{pr_number}")

    diff = get_pr_diff()
    if not diff.strip():
        print("No changes detected in the pull request")
        return

    changed_files = parse_diff_files(diff)
    print(f"Found {len(changed_files)} changed files")

    max_diff_chars = 100000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n\n... (diff truncated)"
        print(f"Diff truncated to {max_diff_chars} characters")

    client = anthropic.Anthropic(api_key=api_key)

    print("Analyzing code with Claude...")
    review_data = analyze_code_with_claude(diff, changed_files, client)

    print("Posting review...")
    post_review(review_data, repo_name, pr_number, changed_files)

    print("AI code review completed successfully!")


if __name__ == "__main__":
    main()
