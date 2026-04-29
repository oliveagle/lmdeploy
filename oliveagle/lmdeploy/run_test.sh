#!/bin/bash

set -e

# Activate the correct virtual environment
source /home/oliveagle/venvs/lmdeploy-dflash/bin/activate

# Run the test
python3 test_accept_rate.py
