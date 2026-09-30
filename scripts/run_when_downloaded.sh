#!/bin/bash
# Wait for the PAT download (xargs) and any running extraction to finish, then (re)run the resumable extraction.
cd "$(dirname "$0")/.."
# Activate your environment first (e.g. conda activate cpg-repr-benchmark).
export PYTHONPATH=src
while pgrep -f "xargs -P 4" >/dev/null || pgrep -f "scripts/extract_reads.py" >/dev/null; do sleep 60; done
python scripts/extract_reads.py --workers 24 > logs_extract2.txt 2>&1
echo "extraction pass finished: $(ls data/reads/*.npz | wc -l) / 253 samples" >> logs_extract2.txt
