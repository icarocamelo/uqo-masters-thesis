#!/bin/bash
python3 prepare_logs.py $1 | xargs -I{} python3 plot.py {}
