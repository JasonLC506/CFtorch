import subprocess
from datetime import datetime
import sys

experiment = sys.argv[1]

var_vals = []
for model in ["CP", "TD", "MultiMF"]:
    for batchsize in [8192]:
        for k in [6,12,24,48]:
            var_vals.append([experiment, model, batchsize, k])

start_from = 0
print "total %d jobs" % len(var_vals)
print "start from %d" % start_from

cnt = 0

log_file = open("log_batch_" + experiment + "_1par", "a")


for i in range(start_from, len(var_vals)):
    qsub_command = """qsub -v experiment={0},MODEL={1},batchsize={2},k={3} experiment_1.pbs""".format(*var_vals[i])
    log_file.write("Job %d\n" % i)
    log_file.write(qsub_command + "\n")
    exit_status = subprocess.call(qsub_command, shell=True, stdout=log_file)
    if exit_status is 1:
        log_file.write("Job {0} failed to submit\n".fotmat(qsub_command))
        break
    cnt += 1
    if cnt >= 99:
        log_file.write("too many jobs end it at %d\n" % i)
        break
log_file.close()
