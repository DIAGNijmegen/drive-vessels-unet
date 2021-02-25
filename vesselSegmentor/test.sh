#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

docker volume create vesselsegmentor-output

docker run --rm \
        --memory=4g \
        -v $SCRIPTPATH/test/:/input/ \
        -v vesselsegmentor-output:/output/ \
        vesselsegmentor

docker run --rm \
        -v vesselsegmentor-output:/output/ \
        python:3.7-slim cat /output/results.json | python -m json.tool

docker run --rm \
        -v vesselsegmentor-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.7-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm vesselsegmentor-output
