import cPickle
import os
import glob
import numpy as np

script        = "/home/reimann/software/python-modules/ps_analysis/hpa/hpa_SourceUniverse_signal_trials.py"
dag_script    = "/home/reimann/software/python-modules/ps_analysis/hpa/source_universe_signal_trials.dag"
outdir        = "/data/user/reimann/2017_10/HPA/sig_trials_source_universe"
submit_script = "/home/reimann/software/python-modules/ps_analysis/hpa/submit.submitter"
wrapper       = "/home/reimann/software/python-modules/ps_analysis/hpa/wrapper.sh"
retry         = 1

infile_signal = "/data/user/reimann/2017_10/HPA/signal_pool_firesong.cPickle"
infile_background = "/data/user/reimann/2017_10/HPA/bgd_pool_cutoff_2_min_ang_dist_1_cut_on_nspot_None.pickle"
expectation = "/data/user/reimann/2017_10/HPA/check_poissonian/mc_trials/spline_expectation_cutoff_pVal_2.0_min_ang_dist_1.00.pickle"

args = "{script} --infile_signal {infile_signal} --infile_background {infile_background} --expectation {expectation} --outdir {outdir} --n_iter 10000 --infile_source_universe {lumi_file} --density {density}"

lumi_files = glob.glob("/data/user/reimann/2017_10/HPA/SourceUnivers/Luminosity_1e+*.pickle")

with open(dag_script, "w") as of:
    for lumi_file in lumi_files: 
        for density in np.logspace(-12, -2, 4*7+1):
            lumi_name = os.path.basename(lumi_file)
            job_name = "sig_trials.{lumi_name}.{density}".format(**locals()).replace("+", "p").replace(".", "_")
            curr_args = args.format(**locals())

            of.write('JOB {job_name} {submit_script}\n'.format(**locals()))
            of.write('VARS {job_name} MEMORY="1800" DISK="2000" PRIORITY="1"\n'.format(**locals()))
            of.write('VARS {job_name} NAME="{job_name}"\n'.format(**locals()))
            of.write('VARS {job_name} SCRIPT="{wrapper}" PARAMS="{curr_args}"\n'.format(**locals()))
            of.write('Retry {job_name} {retry}\n'.format(**locals()))
