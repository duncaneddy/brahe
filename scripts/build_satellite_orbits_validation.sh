#!/usr/bin/env bash

# This script is used to build the satellite orbits validation files.
# These are independent programs that are used to generate data to validate
# the brahe implementation.
#
# Specifically, these are used to validate algorithms and implementations from
# the book "Satellite Orbits: Models, Methods, and Applications" by Montenbruck
# and Gill.

# Directory of build script
SCRIPT_DIR="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)" || return
export ROOT_DIR="$SCRIPT_DIR/../validation/satellite_orbits"
export BUILD_DIR="$SCRIPT_DIR/../build"

function return_home() {
    pushd -0 && dirs -c
}

function build_clean() {
    echo "Cleaning build..."
    rm -rf $BUILD_DIR

}

function build_validation() {

    # Set the working directory to the root of the repository
    cd $ROOT_DIR

    # Build the satellite orbits validation programs
    echo "Building satellite orbits validation programs"
    mkdir -p $BUILD_DIR

    # Build the satellite orbits validation programs
    for file in ./*.cpp; do
        filename=$(basename -- "$file")
        filename="${filename%.*}"
        cpp_src=$(find "$ROOT_DIR"/src/*.cpp)
        echo "Building $filename"
        g++ -I../../validation/satellite_orbits/src $cpp_src $file -o $BUILD_DIR/$filename
        chmod +x $BUILD_DIR/$filename
    done
}

case ${1:-all} in
    build)
        build_validation
        ;;
    clean)
        build_clean
        ;;
    *)
        build_validation || return_home
        build_clean || return_home
        ;;
esac
