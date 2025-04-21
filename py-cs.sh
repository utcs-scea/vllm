#!/bin/bash
# find -name "*.py"     is different from   find -name *.py
rm -f tags
rm -f cscope.files cscope.out cscope.in.out cscope.po.out

ctags -R --python-kinds=-i \
    --exclude=docs \
    --exclude=./deps \
    --exclude=./buildkite \
    --exclude=tests 
find . \
    -path ./docs -prune -o \
    -path ./.deps -prune -o \
    -path ./.buildkite -prune -o \
    -path ./tests -prune -o \
    -name "test_*" -prune -or -name "*.h" -or -name "*.hpp" -or -name "*.hh" -or -name "*.c" -or -name "*.cc" -or -name "*.cxx" -or -name "*.py" -or -name "*.cpp" -or -name "*.cu" -or -name "SConstruct" -or -name "SConscript" -or -name "*.def" > cscope.files
cscope -bkq -i cscope.files
