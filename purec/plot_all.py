import sys
import os

from matplotlib import pyplot as plt
import numpy as np


def main(dirname: str):
    for fname in [os.path.join(dirname, fname)
                  for fname in os.listdir(dirname) if fname.endswith(".dat")]:
        plt.plot(np.fromfile(fname, dtype=np.float64))
        plt.savefig(fname + ".png")
        plt.clf()


if __name__ == "__main__":
    dirname = os.path.join(os.path.abspath(os.curdir), "cmyriad_dat/")
    if len(sys.argv) == 2:
        dirname = sys.argv[1]
    print(dirname)
    main(dirname)
