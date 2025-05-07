import argparse
import os
import sys
import filecmp

def compare_dirs(dir1, dir2):
    """
    Return True if dir1 and dir2 are identical (same files, same sub-dirs, same file contents).
    """
    # Compare directory listings
    cmp = filecmp.dircmp(dir1, dir2)

    # Anything only on one side?
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False

    # Compare common files byte-by-byte
    match, mismatch, errors = filecmp.cmpfiles(
        dir1, dir2, cmp.common_files, shallow=False
    )
    if mismatch or errors:
        return False

    # Recursively compare common subdirectories
    for subdir in cmp.common_dirs:
        if not compare_dirs(
            os.path.join(dir1, subdir),
            os.path.join(dir2, subdir)
        ):
            return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Deep-compare two folders (all nested files/subfolders)."
    )
    parser.add_argument("folder1", help="First folder path")
    parser.add_argument("folder2", help="Second folder path")
    args = parser.parse_args()

    d1 = args.folder1
    d2 = args.folder2

    if not os.path.isdir(d1):
        print(f"Error: '{d1}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(d2):
        print(f"Error: '{d2}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    identical = compare_dirs(d1, d2)
    if identical:
        print("✅ The two directories are identical.")
        sys.exit(0)
    else:
        print("❌ The two directories differ.")
        sys.exit(2)

if __name__ == "__main__":
    main()
