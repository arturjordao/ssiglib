#!/bin/bash

SSF_DIR=`pwd`
BUILD_DIR=$SSF_DIR/build

function build()
{
  mkdir $BUILD_DIR && cd $BUILD_DIR
  cmake -DBUILD_TESTS=OFF $SSF_DIR
  make -j2
}

function test()
{
  mkdir $BUILD_DIR && cd $BUILD_DIR
  cmake -DBUILD_TESTS=ON $SSF_DIR
  make ssf_core -j3
  make test_core
  make ssf_configuration -j3
  make test_configuration
  make test -j3
}

case $TASK in
  build ) build;;
  test ) test;;
esac
