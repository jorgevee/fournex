# Contributing to Fournex

Thanks for your interest in contributing. This guide covers how to set up locally, run the tests, and submit changes.

## Local setup

```bash
git clone https://github.com/fournex/fournex.git
cd fournex

# Install the Python package in editable mode
pip install -e backend/python

# Verify everything works
frx doctor
frx smoke-test
```

Python 3.10+ is required. A CUDA GPU is optional — all tests and the analysis pipeline run on CPU.

## Running tests

```bash
pytest backend/tests/python/
```

All 97 tests should pass. The suite covers bottleneck classification, IR mappers, CLI collection, recommendations, and storage.

## Frontend (optional)

```bash
cd frontend
npm install
npm run dev   # dev server at http://localhost:3000
npm run build # production build check
```

The backend API is needed for the `/analyze` page:

```bash
cd backend
uvicorn api:app --reload  # runs at http://localhost:8000
```

## Making changes

- Open an issue first for anything beyond a small bug fix — this avoids duplicate work.
- Keep PRs focused. One logical change per PR.
- Make sure `frx smoke-test` and `pytest backend/tests/python/` pass before submitting.
- Do not add dependencies without discussion.

## Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your changes.
3. Run `frx smoke-test` and `pytest backend/tests/python/`.
4. Open a PR against `main` with a clear description of what changed and why.

## Reporting bugs

Open a GitHub issue with:
- OS and Python version (`frx doctor` output)
- The command you ran
- The full error output
