#!/usr/bin/env bash

./build.sh

docker save vesselsegmentor | gzip -c > vesselSegmentor.tar.gz
