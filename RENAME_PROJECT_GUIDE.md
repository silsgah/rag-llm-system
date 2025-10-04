# Project Renaming Guide

## Overview
This guide helps you rename the `llm_engineering` package to your preferred name for a fresh GitHub repository.

---

## Recommended New Names

Consider these professional alternatives:
- `rag_system` - Simple, descriptive
- `llm_twin` - Matches the project concept
- `intelligent_assistant` - Professional
- `knowledge_assistant` - Descriptive
- `rag_pipeline` - Technical
- `ai_engineering` - Broader scope

**For this guide, we'll use: `rag_system`** (replace with your choice)

---

## Automated Renaming Script

Save this as `rename_project.py`:

```python
#!/usr/bin/env python3
"""
Automated project renaming script.
Renames llm_engineering to your preferred package name.
"""

import os
import re
import shutil
from pathlib import Path

OLD_NAME = "llm_engineering"
NEW_NAME = "rag_system"  # ← CHANGE THIS

OLD_NAME_UPPER = OLD_NAME.upper()
NEW_NAME_UPPER = NEW_NAME.upper()
OLD_NAME_TITLE = OLD_NAME.replace("_", " ").title()
NEW_NAME_TITLE = NEW_NAME.replace("_", " ").title()

def rename_directory():
    """Rename main package directory."""
    old_path = Path(OLD_NAME)
    new_path = Path(NEW_NAME)

    if old_path.exists():
        print(f"✓ Renaming directory: {old_path} → {new_path}")
        shutil.move(str(old_path), str(new_path))
    else:
        print(f"⚠ Directory {old_path} not found")

def update_file_content(file_path: Path):
    """Update import statements and references in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace import statements
        content = re.sub(
            rf'\bfrom {OLD_NAME}\b',
            f'from {NEW_NAME}',
            content
        )
        content = re.sub(
            rf'\bimport {OLD_NAME}\b',
            f'import {NEW_NAME}',
            content
        )

        # Replace string literals
        content = content.replace(f'"{OLD_NAME}"', f'"{NEW_NAME}"')
        content = content.replace(f"'{OLD_NAME}'", f"'{NEW_NAME}'")

        # Replace uppercase versions
        content = content.replace(OLD_NAME_UPPER, NEW_NAME_UPPER)

        # Replace title case versions
        content = content.replace(OLD_NAME_TITLE, NEW_NAME_TITLE)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def scan_and_update():
    """Scan all Python files and update references."""
    python_files = [
        *Path('.').glob('**/*.py'),
        *Path('.').glob('**/*.toml'),
        *Path('.').glob('**/*.yaml'),
        *Path('.').glob('**/*.yml'),
        *Path('.').glob('**/*.md'),
        *Path('.').glob('**/*.txt'),
    ]

    # Exclude virtual environments and git
    exclude_dirs = {'.venv', 'venv', '.git', '__pycache__', '.pytest_cache',
                   'node_modules', '.zenml', 'outputs'}

    python_files = [
        f for f in python_files
        if not any(excluded in f.parts for excluded in exclude_dirs)
    ]

    updated_count = 0
    for file_path in python_files:
        if update_file_content(file_path):
            print(f"  ✓ Updated: {file_path}")
            updated_count += 1

    print(f"\n✓ Updated {updated_count} files")

def main():
    print("="*60)
    print(f"PROJECT RENAMING: {OLD_NAME} → {NEW_NAME}")
    print("="*60)

    response = input(f"\nRename '{OLD_NAME}' to '{NEW_NAME}'? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    print("\n1. Backing up pyproject.toml...")
    shutil.copy('pyproject.toml', 'pyproject.toml.backup')

    print("\n2. Scanning and updating file contents...")
    scan_and_update()

    print("\n3. Renaming main directory...")
    rename_directory()

    print("\n" + "="*60)
    print("✓ RENAMING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test imports: python -c 'import rag_system'")
    print("2. Run tests: poetry run pytest")
    print("3. Check git status: git status")
    print("4. Commit changes: git add . && git commit -m 'Rename project'")
    print("\nBackup saved: pyproject.toml.backup")

if __name__ == "__main__":
    main()
```

---

## Manual Renaming Steps

If you prefer manual control:

### Step 1: Rename Directory
```bash
mv llm_engineering rag_system
```

### Step 2: Update pyproject.toml
```toml
[tool.poetry]
name = "rag-system"  # ← Change this
version = "0.1.0"
description = "Production-ready RAG system with LLM fine-tuning"  # ← Update
authors = ["Your Name <your.email@example.com>"]  # ← Change

[tool.poetry.dependencies]
python = "~3.11"
# ... rest stays same
```

### Step 3: Update Import Statements

**Files to update:**
- `tools/rag.py`
- `tools/run.py`
- `tools/ml_service.py`
- `tools/data_warehouse.py`
- `steps/**/*.py`
- `pipelines/**/*.py`
- `tests/**/*.py`

**Find and replace:**
```python
# Old:
from llm_engineering.application.rag import ContextRetriever
from llm_engineering.settings import settings

# New:
from rag_system.application.rag import ContextRetriever
from rag_system.settings import settings
```

**Bulk replace command:**
```bash
# Linux/Mac
find . -type f -name "*.py" -exec sed -i '' 's/llm_engineering/rag_system/g' {} +

# Or use grep + sed
grep -rl "llm_engineering" . --include="*.py" | xargs sed -i 's/llm_engineering/rag_system/g'
```

### Step 4: Update Configuration Files

**Files to check:**
- `pyproject.toml`
- `configs/*.yaml`
- `.env.example`
- `Dockerfile`
- `docker-compose.yml`
- `README.md`

### Step 5: Reinstall Package
```bash
poetry install
```

### Step 6: Test
```bash
# Test imports
python -c "import rag_system; print('✓ Import successful')"

# Test RAG
poetry poe call-rag-retrieval-module

# Run tests
poetry run pytest
```

---

## Files That Reference Package Name

### Python Files
- ✅ `tools/*.py` - All tool scripts
- ✅ `steps/**/*.py` - ZenML steps
- ✅ `pipelines/**/*.py` - ZenML pipelines
- ✅ `tests/**/*.py` - Test files
- ✅ `rag_system/**/*.py` - Package itself

### Config Files
- ✅ `pyproject.toml` - Package name
- ✅ `.env.example` - Documentation
- ✅ `README.md` - Documentation
- ⚠️ `configs/*.yaml` - May have references

### Infrastructure
- ✅ `Dockerfile` - COPY and pip install commands
- ✅ `docker-compose.yml` - Service names
- ⚠️ `.github/workflows/*.yml` - CI/CD paths

---

## Verification Checklist

After renaming:

- [ ] Directory renamed: `llm_engineering` → `rag_system`
- [ ] `pyproject.toml` updated
- [ ] All imports updated (check with `grep -r "llm_engineering"`)
- [ ] Package reinstalled: `poetry install`
- [ ] Import works: `python -c "import rag_system"`
- [ ] RAG works: `poetry poe call-rag-retrieval-module`
- [ ] Tests pass: `poetry run pytest`
- [ ] No references left: `grep -r "llm_engineering" . --exclude-dir=.git`

---

## Rollback Plan

If issues occur:

```bash
# Restore backup
cp pyproject.toml.backup pyproject.toml

# Rename back
mv rag_system llm_engineering

# Reinstall
poetry install
```

---

## Git Best Practices

```bash
# Create a new branch for renaming
git checkout -b rename-project

# Make changes
python rename_project.py

# Review changes
git status
git diff

# Commit
git add .
git commit -m "refactor: Rename llm_engineering to rag_system"

# Test before merging
poetry run pytest

# Merge to main
git checkout main
git merge rename-project
```

---

## Recommended Project Names by Use Case

| Use Case | Suggested Name | Description |
|----------|---------------|-------------|
| General RAG | `rag_system` | Simple, clear |
| AI Assistant | `ai_assistant` | User-facing |
| Knowledge Base | `knowledge_engine` | Professional |
| LLM Twin | `llm_twin` | Matches concept |
| Enterprise | `enterprise_ai` | Business-focused |
| Research | `research_assistant` | Academic |
| Custom | `{your_company}_ai` | Branded |

---

**Ready to rename?**

1. Choose your new name
2. Update `NEW_NAME` in the script
3. Run: `python rename_project.py`
4. Test everything
5. Commit changes

---

Generated: 2025-10-04
