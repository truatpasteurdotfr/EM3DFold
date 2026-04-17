import shutil


def getlines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()


def extract_lines_by_ca(lines):
    return [line for line in lines if line.startswith("ATOM") and line[12:16] == " CA "]


def writelines(filename, lines):
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip("\n") + "\n")


def extract_lines_by_res_idx(lines, res_idxs):
    res_idx_set = set(res_idxs)
    ret = []
    for line in lines:
        if line.startswith("ATOM"):
            res_idx = int(line[22:26])
            if res_idx in res_idx_set:
                ret.append(line)
    return ret


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
        return True
    except Exception:
        return False

