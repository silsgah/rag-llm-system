# Migration Guide: Poetry → UV

## Current Status
- ✅ UV v0.8.17 installed
- ✅ Python 3.11.8
- ⚠️ Poetry pyproject.toml format
- ⚠️ Poe the Poet task runner

## Migration Options

### Option 1: Full Migration (Recommended)
**Time**: ~15 minutes
**Risk**: Low (can revert easily)
**Benefits**: Maximum performance, modern tooling

### Option 2: UV for Dependencies Only
**Time**: ~5 minutes
**Risk**: Very Low
**Benefits**: Faster installs, keep familiar Poetry commands

---

## Full Migration Steps

### Step 1: Convert Poetry lock to UV format
```bash
# Backup current setup
cp poetry.lock poetry.lock.backup
cp pyproject.toml pyproject.toml.backup

# Initialize UV from existing pyproject.toml
uv sync
```

### Step 2: Update pyproject.toml build system
Replace:
```toml
[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

With:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-engineering"
version = "0.1.0"
description = ""
authors = [{name = "Paul Iusztin", email = "p.b.iusztin@gmail.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = "~=3.11"
dependencies = [
    "zenml[server]==0.74.0",
    "pymongo>=4.6.2",
    "click>=8.0.1",
    # ... (convert all dependencies)
]

[project.optional-dependencies]
dev = ["ruff>=0.4.9", "pre-commit>=3.7.1", "pytest>=8.2.2"]
aws = ["sagemaker>=2.232.2", "s3fs>2022.3.0", ...]
```

### Step 3: Update Poe the Poet tasks
Replace `poetry run` with `uv run` in all tasks:

```toml
[tool.poe.tasks]
run-digital-data-etl = "uv run python -m tools.run --run-etl --no-cache"
call-rag-retrieval-module = "uv run python -m tools.rag"
# ... etc
```

### Step 4: Update commands in README/scripts
```bash
# Old
poetry install
poetry poe run-training-pipeline

# New
uv sync
uv run poe run-training-pipeline
# or directly
uv run python -m tools.run --run-training
```

---

## Hybrid Approach (Minimal Changes)

### Just use UV for installation speed:

```bash
# Instead of: poetry install
uv pip install -e .

# Keep using Poetry commands
poetry poe run-digital-data-etl
```

---

## Migration Script

Run this to do full migration:

```bash
#!/bin/bash
set -e

echo "Starting Poetry → UV migration..."

# 1. Backup
cp pyproject.toml pyproject.toml.poetry.backup
cp poetry.lock poetry.lock.backup

# 2. Convert dependencies
echo "Converting pyproject.toml..."
# (Use Python script to convert - see below)

# 3. Sync with UV
echo "Syncing dependencies with UV..."
uv sync --all-extras

# 4. Test
echo "Testing installation..."
uv run python -c "import zenml; import torch; print('✓ Installation successful')"

echo "✓ Migration complete!"
```

---

## Rollback Plan

If issues occur:
```bash
# Restore Poetry setup
cp pyproject.toml.poetry.backup pyproject.toml
cp poetry.lock.backup poetry.lock
poetry install

# Remove UV cache
rm -rf .venv
rm uv.lock
```

---

## Performance Comparison

**Initial Install:**
- Poetry: ~5-10 minutes
- UV: ~30-60 seconds

**Adding package:**
- Poetry: ~1-2 minutes
- UV: ~2-5 seconds

**Lock file update:**
- Poetry: ~30-60 seconds
- UV: ~1-3 seconds

---

## Compatibility Notes

1. **Poe the Poet**: Fully compatible with UV (just change `poetry run` → `uv run`)
2. **ZenML**: Works perfectly with UV
3. **AWS SageMaker**: No issues
4. **Docker**: Update Dockerfile to use UV
5. **CI/CD**: Update GitHub Actions

---

## Recommendation

**For this project: Option 2 (Hybrid) initially**

Why?
- Project is already working with Poetry
- Minimal disruption
- Can fully migrate later if desired
- Get 90% of UV benefits with 10% of the work

To use hybrid:
```bash
# Install dependencies faster
uv pip sync poetry.lock

# Keep using familiar commands
poetry poe call-rag-retrieval-module
```
