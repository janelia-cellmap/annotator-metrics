. /misc/lsf/conf/profile.lsf
export PATH="/groups/scicompsoft/home/ackermand/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate annotator-metrics
/groups/scicompsoft/home/ackermand/Programming/annotator-metrics/monitor.py
