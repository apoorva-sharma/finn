from os import walk
from os import path
from os import getcwd
import subprocess
import sys

base_path = sys.argv[1];
(dirpath, _, filenames) = walk(base_path).next()
for filename in filenames:
    if filename[0:12] == "G_epoch95img" or filename[0:12] == "Z2_epoch0img":
        full_path = path.join(dirpath, filename)
        dest_path = path.join(dirpath, filename[12:])
        cmd = "mv " + full_path + " " + dest_path
        print(cmd)
        subprocess.Popen(cmd.split())
