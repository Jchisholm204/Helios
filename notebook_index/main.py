import os

VAULT_SOURCE_DIR = "sample_vault"


def print_dir(base_dir: str, indent: int = 1):
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        return
    dirs = os.listdir(base_dir)
    # print(os.path.basename(base_dir))
    for dir in dirs:
        print("|"*indent + f"> {dir}")
        if os.path.isdir(os.path.join(base_dir, dir)):
            print_dir(os.path.join(base_dir, dir), indent+1)


def main():
    wd = os.path.join(os.getcwd(), VAULT_SOURCE_DIR)
    print(f"> {os.path.basename(wd)}")
    print_dir(wd)


if __name__ == "__main__":
    main()
