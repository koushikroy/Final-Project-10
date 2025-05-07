import os
import argparse
import zipfile
from pathlib import Path

def zip_subfolders(root_path: Path):
    """
    For each subfolder in root_path, create a zip archive of its contents
    named <subfolder_name>.zip in the root_path.
    """
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: '{root_path}' is not a valid directory.")
        return

    # Iterate over each item in the given directory
    for item in root_path.iterdir():
        if item.is_dir():
            zip_name = root_path / f"{item.name}.zip"
            print(f"Creating archive: {zip_name}")
            # Create zip archive
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the directory tree
                for folder_root, _, files in os.walk(item):
                    for file in files:
                        file_path = Path(folder_root) / file
                        # Compute the archive name relative to the subfolder
                        arcname = file_path.relative_to(item)
                        zipf.write(file_path, arcname)
            print(f"Finished: {zip_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Zip each subfolder in a given directory into separate archives."
    )
    parser.add_argument(
        'folder',
        type=Path,
        help='Path to the directory containing subfolders to zip.'
    )
    args = parser.parse_args()
    zip_subfolders(args.folder)

if __name__ == '__main__':
    main()
