import subprocess
from datetime import datetime
import sys

#experiment = sys.argv[1]
experiment = "experiment.py"
#data_sets = ["CNN", "foxnews", "period_foxnews", "period_foxnews_nolike"]
data_sets = ["CNN_nolike"]

var_vals = []
for model in ["NTN"]:
    for batchsize in [32768]:
        for k in [16]:
            for k2 in [16]:
                for data in data_sets:
                    var_vals.append([experiment, model, batchsize, k, k2, data])

start_from = 0
print "total %d jobs" % len(var_vals)
print "start from %d" % start_from

cnt = 0

log_file = open("log_batch_" + experiment + "_2par", "a")


for i in range(start_from, len(var_vals)):
    qsub_command = """qsub -v experiment={0},data={5},MODEL={1},batchsize={2},k={3},k2={4} experiment_2.pbs""".format(*var_vals[i])
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
