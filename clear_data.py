#!/usr/bin/env python3
"""
Clear all application data - resets the app to fresh state
Deletes:
- Uploaded audio files
- Processed/segmented audio
- Training datasets
- Trained models
- Cache files
"""
import shutil
from pathlib import Path
import sys


def clear_directory(directory: Path, description: str) -> bool:
    """Clear all contents of a directory"""
    if not directory.exists():
        print(f"  ⚠️  {description} doesn't exist, skipping...")
        return True

    try:
        # Count files before deletion
        file_count = sum(1 for _ in directory.rglob('*') if _.is_file())

        if file_count == 0:
            print(f"  ✓ {description} is already empty")
            return True

        # Delete all contents but keep the directory
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        print(f"  ✓ Cleared {description} ({file_count} files deleted)")
        return True
    except Exception as e:
        print(f"  ✗ Error clearing {description}: {e}")
        return False


def main():
    print("=" * 60)
    print("🗑️  CLEAR ALL APPLICATION DATA")
    print("=" * 60)
    print()
    print("This will DELETE ALL of the following:")
    print("  • Uploaded audio files (backend/data/uploads/)")
    print("  • Processed audio segments (backend/data/processed/)")
    print("  • Training datasets (backend/data/datasets/)")
    print("  • Trained voice models (backend/trained_models/)")
    print("  • Generated audio (backend/data/generated/)")
    print("  • Cache files (backend/cache/)")
    print()
    print("⚠️  WARNING: This action CANNOT be undone!")
    print()

    # Ask for confirmation
    response = input("Type 'yes' to confirm deletion: ").strip().lower()

    if response != 'yes':
        print("\n❌ Cancelled. No files were deleted.")
        sys.exit(0)

    print("\n🗑️  Starting cleanup...\n")

    # Define directories to clear
    base_dir = Path(__file__).parent / "backend"
    directories = [
        (base_dir / "data" / "uploads", "Uploads"),
        (base_dir / "data" / "processed", "Processed audio"),
        (base_dir / "data" / "datasets", "Training datasets"),
        (base_dir / "data" / "generated", "Generated audio"),
        (base_dir / "trained_models", "Trained models"),
        (base_dir / "cache", "Cache files"),
    ]

    success = True
    for directory, description in directories:
        if not clear_directory(directory, description):
            success = False

    print()
    if success:
        print("✅ All data cleared successfully!")
        print("The application has been reset to a fresh state.")
    else:
        print("⚠️  Some errors occurred during cleanup.")
        print("Please check the messages above.")

    print()


if __name__ == "__main__":
    main()
