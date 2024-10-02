# i=11
# start=$(($i*100))
# end=$((100*$i+100))
start=1140
end=1155
cmd="python backward_generation.py --start ${start} --end ${end}"

$cmd
# jbsub -q x86_24h -cores 1x8+1 -require a100_80gb -mem 100G ${cmd}