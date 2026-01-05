import os

VAULT_SOURCE_DIR = "sample_vault"


class Directory:
    path: str
    subdirs: list
    files: list

    def __init__(self, name: str):
        self.path = name
        self.subdirs = []
        self.files = []

    def print(self, indent: int = 0):
        print("  "*indent + os.path.basename(self.path))
        print("    "*indent + f"->{self.files}")
        for subdir in self.subdirs:
            subdir.print(indent=indent+1)


def scan_children(base_dir: str) -> Directory:
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        return None
    found: Directory = Directory(base_dir)
    dirs = os.listdir(base_dir)
    for dir in dirs:
        sdir = os.path.join(base_dir, dir)
        if os.path.isdir(sdir):
            found.subdirs.append(scan_children(sdir))
        else:
            found.files.append(dir)
    return found


def main():
    wd = os.path.join(os.getcwd(), VAULT_SOURCE_DIR)
    scan_children(wd).print()


if __name__ == "__main__":
    main()
