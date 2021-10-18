#!/usr/bin/env bash

./build.sh

docker save vesselsegmentation | gzip -c > VesselSegmentation.tar.gz
