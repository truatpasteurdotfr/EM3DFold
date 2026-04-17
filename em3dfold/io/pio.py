import numpy as np

def read_p(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    coords = []
    dens = []
    for line in lines:
        if line.startswith("ATOM"):
            xyz = [float(x) for x in [line[i:i+8] for i in [30, 38, 46]]]
            d = float(line[54:60])
            coords.append(xyz)
            dens.append(d)
    coords = np.asarray(coords)
    dens = np.asarray(dens)
    return coords, dens


def write_p(filename, coords, dens):
    with open(filename, 'w') as f:
        for k, (coord, d) in enumerate(zip(coords, dens)):
            f.write("ATOM  {:>5d}  CA  GLY A{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:>6.2f}{:>6.2f}              \n".format(
                k+1 if k+1 <= 99999 else 99999,
                k+1 if k+1 <= 9999 else 9999,
                coord[0],
                coord[1],
                coord[2],
                d,
                d,
            ))
            f.write("TER\n")

