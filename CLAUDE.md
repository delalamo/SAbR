# CLAUDE.md

Guidelines for AI assistants working on this codebase.

## Pre-commit Hooks

Always run pre-commit hooks before pushing code:

```bash
pre-commit run --all-files
```

If pre-commit is not installed, install it first:

```bash
pip install pre-commit
pre-commit install
```

## Type Hints

Avoid using type hints outside of function definitions. Type hints should only be used in function signatures (parameters and return types), not for variable assignments or class attributes.

**Do this:**
```python
def process_sequence(sequence: str, threshold: float = 0.5) -> list:
    result = []
    count = 0
    # ...
    return result
```

**Don't do this:**
```python
def process_sequence(sequence: str, threshold: float = 0.5) -> list:
    result: list = []
    count: int = 0
    # ...
    return result
```
