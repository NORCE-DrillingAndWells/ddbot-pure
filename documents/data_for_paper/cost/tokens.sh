#!/bin/bash
name="no_pruning_20250516"
echo "${name}"
grep "prompt_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee ./documents/data_for_paper/cost/${name}.txt
grep "completion_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee -a ./documents/data_for_paper/cost/${name}.txt

name="baseline_20250518"
echo "${name}"
grep "prompt_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee ./documents/data_for_paper/cost/${name}.txt
grep "completion_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee -a ./documents/data_for_paper/cost/${name}.txt

name="cot_20250518"
echo "${name}"
grep "prompt_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee ./documents/data_for_paper/cost/${name}.txt
grep "completion_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee -a ./documents/data_for_paper/cost/${name}.txt

name="pc_20250515"
echo "${name}"
grep "prompt_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee ./documents/data_for_paper/cost/${name}.txt
grep "completion_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee -a ./documents/data_for_paper/cost/${name}.txt

name="pc_cot_20250515"
echo "${name}"
grep "prompt_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee ./documents/data_for_paper/cost/${name}.txt
grep "completion_tokens" ./tasks/3Tests_${name}/00001_demo/record.log | tee -a ./documents/data_for_paper/cost/${name}.txt
