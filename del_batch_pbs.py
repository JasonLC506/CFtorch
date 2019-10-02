import subprocess
import sys

log_file = sys.argv[1]
job_to_pass = set(sys.argv[2:])

with open(log_file, "r") as logf:
    for line in logf:
        if "production.int.aci.ics.psu.edu" in line:
            if line.rstrip("\n") in job_to_pass:
                continue 
            cmd = """qdel %s""" % (line.rstrip("\n"))
            print cmd
            subprocess.call(cmd, shell=True)
        else:
            print line
