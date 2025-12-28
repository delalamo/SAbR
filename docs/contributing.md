# Contributing

We welcome contributions to SAbR! This guide will help you get started.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue at:
[GitHub Issues](https://github.com/delalamo/SAbR/issues)

When reporting bugs, please include:

- A minimal reproducible example
- The version of SAbR you're using
- Your Python version and operating system
- The full error traceback

## Development Setup

1. Clone the repository with submodules:

    ```bash
    git clone --recursive https://github.com/delalamo/SAbR.git
    cd SAbR
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install development dependencies:

    ```bash
    pip install -e ".[test]"
    pip install pre-commit
    ```

4. Install pre-commit hooks:

    ```bash
    pre-commit install
    ```

## Running Tests

Run the test suite with pytest:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=sabr --cov-report=html
```

## Code Style

SAbR uses the following tools for code quality:

| Tool | Purpose |
|------|---------|
| Black | Code formatting (line length 80) |
| isort | Import sorting (black profile) |
| flake8 | Linting |

The pre-commit hooks will automatically check and format your code.

To manually run the formatters:

```bash
black src/ tests/
isort src/ tests/
```

## Building Documentation

To build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

To build static HTML:

```bash
mkdocs build
```

The built documentation will be in `site/`.

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

Please ensure your PR:

- Includes tests for new functionality
- Updates documentation as needed
- Follows the existing code style
- Has a clear description of the changes
