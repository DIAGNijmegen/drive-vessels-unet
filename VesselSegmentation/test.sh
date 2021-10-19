#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

docker volume create vesselsegmentation-output-$VOLUME_SUFFIX

# run the forward pass and store the outputs in a temporary Docker volume
docker run --rm \
        --gpus=all \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v vesselsegmentation-output-$VOLUME_SUFFIX:/output/ \
        vesselsegmentation

# compare the outputs in the Docker volume with the outputs in ./test/expected_output/
docker run --rm \
        -v vesselsegmentation-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output/:/expected_output/ \
        biocontainers/simpleitk:v1.0.1-3-deb-py3_cv1 python3 -c """
import SimpleITK as sitk

output = sitk.ReadImage('/output/images/01_test.tif')
expected_output = sitk.ReadImage('/expected_output/images/01_test.tif')

label_filter = sitk.LabelOverlapMeasuresImageFilter()
label_filter.Execute(output, expected_output)
dice_score = label_filter.GetDiceCoefficient()

if dice_score == 1.0:
    print('Test passed!')
else:
    print('Test failed!')
"""

docker volume rm vesselsegmentation-output-$VOLUME_SUFFIX
