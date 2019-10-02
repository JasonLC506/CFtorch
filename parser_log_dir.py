import numpy as np
import re
import sys
import os

dir_name = sys.argv[1]
for fname in os.listdir(dir_name):
    file_name = os.path.join(dir_name, fname)
    if os.path.isdir(file_name):
        continue
    ppl = []
    pattern = re.compile(r'^Epoch')
    with open(file_name, "r") as f:
        for line in f:
            if not pattern.match(line):
                continue
            records = (line.rstrip("\n")).split(" ")
            #print records
            ppl.append(map(float, [records[-7], records[-3]]))
    ppl = np.array(ppl)
    print file_name
    if ppl.shape[0] >= 1:
        print "min train loss", np.min(ppl[:,0])
        print "min valid loss", np.min(ppl[:,1])
    else:
        print "no results"
        
