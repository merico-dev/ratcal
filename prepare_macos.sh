#!/bin/sh

PREFIX=`brew --prefix suite-sparse`
export LDFLAGS="-L$PREFIX/lib"
export CPPFLAGS="-I$PREFIX/include"
