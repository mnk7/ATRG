#!/bin/bash

valgrind --tool=massif ./build/atrg_test --heap=yes --stack=yes --detailed-freq=1 --max-snapshots=100000
