import os
from io import TextIOWrapper

VAULT_SOURCE_DIR = "sample_vault"


class Directory:
    path: str
    subdirs: list
    files: list

    def __init__(self, name: str):
        self.path = name
        self.subdirs = []
        self.files = []

    def get_name(self) -> str:
        return os.path.basename(self.path)

    def print(self, indent: int = 0):
        print("  "*indent + self.get_name())
        print("    "*indent + f"->{self.files}")
        for subdir in self.subdirs:
            subdir.print(indent=indent+1)

    def subdir_depth(self) -> int:
        max_depth = 0
        if len(self.subdirs) == 0:
            return 1
        for sd in self.subdirs:
            if sd.subdir_depth() > max_depth:
                # Return Depth of subdir + this dir
                max_depth = sd.subdir_depth()
        return max_depth + 1

    def subdir_files(self) -> int:
        sd_files = len(self.files)
        for sd in self.subdirs:
            sd_files += sd.subdir_files()
        return sd_files

    def filter(self, filter=['.md', '.pdf', '.txt'], ignorelist=['.venv', '.pio']):
        # Filter through self files
        filtered_files = []
        for file in self.files:
            if os.path.splitext(file)[1] in filter:
                filtered_files.append(file)
        self.files = filtered_files
        filtered_dirs = []
        for sd in self.subdirs:
            if sd.get_name() in ignorelist:
                continue
            sd.filter(filter)
            if sd.subdir_files() > 0:
                filtered_dirs.append(sd)
        self.subdirs = filtered_dirs


def scan_children(base_dir: str, max_depth: int = 999) -> Directory:
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        return None
    if max_depth == 0:
        return None
    max_depth = max_depth - 1
    found: Directory = Directory(base_dir)
    dirs = os.listdir(base_dir)
    for dir in dirs:
        sdir = os.path.join(base_dir, dir)
        if os.path.isdir(sdir):
            children = scan_children(sdir, max_depth)
            if children is not None:
                found.subdirs.append(children)
        else:
            found.files.append(dir)
    return found


def generate_md(dir: Directory, f: TextIOWrapper = None, indent: int = 1):
    base: bool = f is None
    if base:
        f = open(os.path.join(dir.path, f"{dir.get_name()} Index.md"), "w")
    if dir is None:
        return
    f.write("#"*indent + f" {dir.get_name()}\n")
    for file in dir.files:
        f.write(f"- [[{file}]]\n")
    for d in dir.subdirs:
        generate_md(d, f, indent+1)
    if base:
        f.close()


def main():
    # wd = os.path.join(os.getcwd(), VAULT_SOURCE_DIR)
    wd = os.getcwd()
    dirs = scan_children(wd)
    dirs.filter()
    dirs.print()
    generate_md(dirs)


if __name__ == "__main__":
    main()
