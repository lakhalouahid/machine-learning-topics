#!/bin/bash

check_dir() {
  [ -d $1 ] || mkdir $1
}
check_dir data; check_dir models


[ -f "cifar-10.tar.gz" ] || wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10.tar.gz && tar -xzpf cifar-10.tar.gz -C data
