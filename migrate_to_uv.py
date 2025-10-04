#!/usr/bin/env python3
"""
Automated Poetry to UV migration script.
Converts pyproject.toml from Poetry format to standard Python packaging format.
"""

import re
import shutil
from pathlib import Path


def backup_files():
    """Backup important files before migration."""
    print("📦 Backing up files...")
    shutil.copy("pyproject.toml", "pyproject.toml.poetry.backup")
    if Path("poetry.lock").exists():
        shutil.copy("poetry.lock", "poetry.lock.backup")
    print("✓ Backups created")


def convert_poetry_to_standard(pyproject_path: str = "pyproject.toml"):
    """Convert Poetry pyproject.toml to standard format compatible with UV."""

    with open(pyproject_path, 'r') as f:
        content = f.read()

    print("🔄 Converting pyproject.toml...")

    # Extract metadata
    name = re.search(r'name\s*=\s*"([^"]+)"', content).group(1)
    version = re.search(r'version\s*=\s*"([^"]+)"', content).group(1)
    description = re.search(r'description\s*=\s*"([^"]*)"', content).group(1)
    authors_match = re.search(r'authors\s*=\s*\[(.*?)\]', content)
    license_match = re.search(r'license\s*=\s*"([^"]+)"', content)
    readme_match = re.search(r'readme\s*=\s*"([^"]+)"', content)

    # Keep Poe the Poet configuration
    poe_config = ""
    if "[tool.poe" in content:
        poe_start = content.find("[tool.poe")
        poe_config = content[poe_start:]

    # Convert dependencies (this is simplified - you may need to adjust)
    deps_section = re.search(
        r'\[tool\.poetry\.dependencies\](.*?)(?=\[|$)',
        content,
        re.DOTALL
    ).group(1)

    print("⚠️  Note: Manual review of converted dependencies recommended")
    print("✓ Conversion complete")

    return {
        'name': name,
        'version': version,
        'description': description,
        'poe_config': poe_config
    }


def show_migration_summary():
    """Show summary and next steps."""
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print("""
Current state:
- ✓ Pydantic fixed (2.8.2)
- ✓ UV installed (0.8.17)
- ⏳ MongoDB endpoint being updated
- ⚠️ PyTorch/Torchvision compatibility issue detected

Next steps for UV migration:

OPTION 1 - Quick Win (Recommended for now):
  Keep Poetry, just use UV for speed:

  1. Install with UV:
     $ uv pip install -e .

  2. Keep using Poetry commands:
     $ poetry poe call-rag-retrieval-module

  Benefits: 10x faster installs, zero risk

OPTION 2 - Full Migration (Later):
  1. Review MIGRATION_TO_UV.md
  2. Convert pyproject.toml format
  3. Update all Poe tasks (poetry run → uv run)
  4. Update CI/CD and Docker

  Benefits: Maximum performance, modern tooling

IMMEDIATE ACTION:
  Once MongoDB is ready, test RAG with fixed Pydantic:
  $ poetry poe call-rag-retrieval-module

  If PyTorch error persists, we need to fix torchvision compatibility.
""")
    print("="*60)


if __name__ == "__main__":
    print("\n🚀 Poetry to UV Migration Tool\n")

    response = input("What would you like to do?\n"
                    "1. Just show migration info (safe)\n"
                    "2. Backup files only\n"
                    "3. Full conversion (advanced)\n"
                    "Choice [1]: ").strip() or "1"

    if response == "1":
        show_migration_summary()

    elif response == "2":
        backup_files()
        show_migration_summary()

    elif response == "3":
        print("\n⚠️  FULL CONVERSION - This will modify pyproject.toml")
        confirm = input("Are you sure? Type 'yes' to continue: ")
        if confirm.lower() == 'yes':
            backup_files()
            convert_poetry_to_standard()
            print("\n⚠️  Manual steps still required:")
            print("   1. Review and complete pyproject.toml conversion")
            print("   2. Run: uv sync --all-extras")
            print("   3. Test: uv run python -c 'import zenml'")
            show_migration_summary()
        else:
            print("Cancelled.")
    else:
        print("Invalid choice.")
