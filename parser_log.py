import numpy as np
import re
import sys

file_name = sys.argv[1]
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
print "min train loss", np.min(ppl[:,0])
print "min valid loss", np.min(ppl[:,1])
        
