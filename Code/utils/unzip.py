import argparse
import zipfile
from pathlib import Path

def unzip_archives(input_dir: Path, output_dir: Path):
    """
    For each .zip file in input_dir, extract its contents into a folder
    named after the zip (without extension) inside output_dir.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a valid directory.")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each item in the input directory
    for item in input_dir.iterdir():
        # Process only .zip files
        if item.is_file() and item.suffix.lower() == '.zip':
            # Directory to extract into, named after the zip file (without .zip)
            extract_dir = output_dir / item.stem
            print(f"Extracting '{item.name}' to '{extract_dir}'...")
            # Create the directory if it doesn't exist
            extract_dir.mkdir(parents=True, exist_ok=True)
            # Extract all contents of the zip into the directory
            with zipfile.ZipFile(item, 'r') as zipf:
                zipf.extractall(extract_dir)
            print(f"Finished extracting '{item.name}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract each zip in a given directory into separate folders."
    )
    parser.add_argument(
        'folder',
        type=Path,
        help='Path to the directory containing zip files to extract.'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Directory where extracted folders will be placed. Defaults to input folder.'
    )
    args = parser.parse_args()

    input_dir = args.folder
    output_dir = args.output or args.folder

    unzip_archives(input_dir, output_dir)

if __name__ == '__main__':
    main()
