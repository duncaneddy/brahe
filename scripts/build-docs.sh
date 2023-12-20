#!/bin/bash
#
# Description:
#
#   Script to build project documentation
#
# Usage:
#   $ ./build-docs.sh
#

# Directory of build script
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )" || return
export DOCS_DIR="$SCRIPT_DIR/../docs/"

function return_home() {
    pushd -0 && dirs -c
}

function install_dependencies() {
    pip install -U pip
    echo "Installing dependencies..."
    pip install -U -r "$DOCS_DIR/requirements.txt"
}

function build_docs() {
    echo "Building docs..."
    pushd "$DOCS_DIR" || return
    
    # Build figures

    # Build examples

    # Copy README.md to docs/index.md
    cp ../README.md ./pages/index.md

    # Build docs
    mkdocs build
}

function serve_docs {
    echo "Serving docs..."

    # Serve documents
    mkdocs serve # Add -v for debug if desired

    # Return to source director
    # return_home
}

function publish_docs {
    echo "Beginning Publish Step"

    # Change to English docs directory
    cd "$DOCS_DIR" || exit 1

    # Compile documents
    mkdocs gh-deploy --force || exit 1
}

case ${1:-all} in
    install)
        install_dependencies
        ;;
    test)
        echo "Not Yet Implelemented"
        # Test specific script
        ;;
    build)
        build_docs || return_home
        ;;
    serve)
        serve_docs || return_home
        ;;
    publish)
        publish_docs || return_home
        ;;
    *)
        build_docs || return_home
        serve_docs || return_home
esac