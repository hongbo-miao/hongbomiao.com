#!/usr/bin/env bash
set -e

counter=1
while true; do
    echo "$(date +"%b %e %T") $(hostname) com.hongbomiao.dummy[1]: $counter" | tee -a data/dummy.log
    counter=$((counter+1))
    sleep 3
done
