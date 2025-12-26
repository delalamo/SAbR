#!/usr/bin/env python3
"""AI-powered code review using Anthropic's Claude API."""

import json
import os
import re
import sys
from pathlib import Path

import anthropic
from github import Github


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
        # Match file header: +++ b/path/to/file
        if line.startswith("+++ b/"):
            current_file = line[6:]
            files[current_file] = []
        # Match hunk header: @@ -old,count +new,count @@
        elif line.startswith("@@") and current_file:
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1))
        # Track added/modified lines
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
        "You are an expert code reviewer. Analyze the provided git "
        "diff and provide constructive feedback.\n\n"
        "CRITICAL GUIDELINES:\n"
        "1. **Only report REAL issues** - Do NOT fabricate or invent "
        "problems that don't exist\n"
        "2. **If the code is good, say so** - Not every PR has problems\n"
        "3. **Be specific** - Use exact file paths and line numbers\n"
        "4. **Focus on what matters**: bugs, security, performance, "
        "clear best practice violations\n"
        "5. **Ignore minor style issues** - Don't nitpick formatting\n"
        "6. **When uncertain, don't comment** - Only flag clear issues\n\n"
        "RESPONSE FORMAT - You MUST respond with valid JSON:\n"
        "{\n"
        '    "summary": "Brief 1-2 sentence overall assessment",\n'
        '    "approval": "APPROVE" or "REQUEST_CHANGES",\n'
        '    "comments": [\n'
        "        {\n"
        '            "file": "exact/path/from/diff.py",\n'
        '            "line": <line_number_from_diff>,\n'
        '            "body": "Description of issue and suggested fix"\n'
        "        }\n"
        "    ]\n"
        "}\n\n"
        "CRITICAL RULES FOR COMMENTS:\n"
        "- EVERY issue MUST be in the comments array with file and line\n"
        "- Do NOT describe issues in the summary - put them in comments\n"
        "- The summary should only be a brief overall assessment\n"
        "- Use EXACT file paths as shown in the diff (e.g., src/foo.py)\n"
        "- Use actual line numbers from the + lines in the diff\n"
        "- If no issues, use approval=APPROVE with empty comments array\n"
        "- Only use REQUEST_CHANGES if there are items in comments array"
    )

    user_prompt = (
        "Review this pull request diff. The following files were "
        f"changed:\n{files_context}\n\n"
        f"```diff\n{diff}\n```\n\n"
        "Analyze the changes and respond with JSON. Remember:\n"
        '- Empty "comments" array is correct if the code has no issues\n'
        "- Only flag genuine problems you can specifically identify\n"
        "- Use exact file paths and line numbers from the diff"
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )

    response_text = message.content[0].text

    # Debug: print raw response
    print(f"Raw Claude response:\n{response_text}\n")

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if json_match:
        response_text = json_match.group(1)

    try:
        result = json.loads(response_text.strip())
        print(
            f"Parsed review: approval={result.get('approval')}, "
            f"comments={len(result.get('comments', []))}"
        )
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Raw response: {response_text}")
        # Return a safe default
        return {
            "summary": "Unable to parse AI response. Please review manually.",
            "approval": "APPROVE",
            "comments": [],
        }


def post_review(
    review_data: dict,
    repo_name: str,
    pr_number: int,
    changed_files: dict[str, list[int]],
) -> None:
    """Post the review with file-specific comments using GitHub's review API."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)

    gh = Github(token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Get the latest commit for the review
    commit = pr.get_commits().reversed[0]

    summary = review_data.get("summary", "No summary provided")
    approval = review_data.get("approval", "APPROVE")
    comments = review_data.get("comments", [])

    # Build review body
    if approval == "APPROVE":
        body = f"## :white_check_mark: AI Code Review - Approved\n\n{summary}"
        if not comments:
            body += "\n\n*No issues found. The changes look good!*"
    else:
        body = f"## :warning: AI Code Review - Changes Requested\n\n{summary}"

    body += (
        "\n\n---\n*This review was generated by Claude AI. "
        "Please use your judgment when considering these suggestions.*"
    )

    # Prepare review comments with validated line numbers
    review_comments = []
    print(f"Processing {len(comments)} comments from Claude...")
    for i, comment in enumerate(comments):
        file_path = comment.get("file", "")
        line = comment.get("line", 0)
        comment_body = comment.get("body", "")

        print(f"  Comment {i+1}: {file_path}:{line}")

        # Validate the file exists in the diff
        if file_path not in changed_files:
            print("    -> Skipping: file not in diff")
            print(f"    -> Available files: {list(changed_files.keys())}")
            continue

        # Validate line number - find closest valid line if needed
        valid_lines = changed_files.get(file_path, [])
        if not valid_lines:
            print("    -> Skipping: no valid lines for file")
            continue

        if line not in valid_lines:
            closest = min(valid_lines, key=lambda x: abs(x - line))
            print(f"    -> Line {line} not in diff, using closest: {closest}")
            line = closest

        review_comments.append(
            {
                "path": file_path,
                "line": line,
                "body": comment_body,
            }
        )
        print(f"    -> Added comment at {file_path}:{line}")

    # Determine review event type
    if approval == "REQUEST_CHANGES" and review_comments:
        event = "REQUEST_CHANGES"
    elif approval == "APPROVE":
        event = "APPROVE"
    else:
        event = "COMMENT"

    # Create the review
    if review_comments:
        pr.create_review(
            commit=commit,
            body=body,
            event=event,
            comments=review_comments,
        )
        print(
            f"Posted review with {len(review_comments)} file-specific comments"
        )
    else:
        # No file-specific comments, just post the summary
        pr.create_review(
            commit=commit,
            body=body,
            event=event,
        )
        print("Posted review summary (no file-specific comments)")

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

    pr_number = int(pr_number_str)

    print(f"Starting AI code review for {repo_name}#{pr_number}")

    diff = get_pr_diff()
    if not diff.strip():
        print("No changes detected in the pull request")
        return

    # Parse the diff to understand file structure
    changed_files = parse_diff_files(diff)
    print(f"Found {len(changed_files)} changed files")

    # Truncate very large diffs to avoid token limits
    max_diff_chars = 100000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + "\n\n... (diff truncated due to size)"
        print(f"Diff truncated to {max_diff_chars} characters")

    client = anthropic.Anthropic(api_key=api_key)

    print("Analyzing code with Claude...")
    review_data = analyze_code_with_claude(diff, changed_files, client)

    print("Posting review...")
    post_review(review_data, repo_name, pr_number, changed_files)

    print("AI code review completed successfully!")


if __name__ == "__main__":
    main()
