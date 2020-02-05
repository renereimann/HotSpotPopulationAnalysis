#!/bin/bash
#USAGE wrapper.sh "script -params"

echo "Script $1"
echo "Arguments ${@:2}"

eval `/cvmfs/icecube.opensciencegrid.org/py2-v2/setup.sh`
export PYTHONPATH=/home/reimann/software/python-modules:$PYTHONPATH
export PYTHONPATH=/data/user/reimann/projects/skylab_svn/:$PYTHONPATH
export SKYLABRC=/home/reimann
echo "start script"
python $1 "${@:2}"
retn_code=$?

exit $retn_code
