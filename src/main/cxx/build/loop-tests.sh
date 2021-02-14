#!/bin/bash
TEST_LOOPS=0
while ! [[ "$MYVAR" =~ fail ]]; do
MYVAR=`./test-results.out --use-colour yes -a`
((TEST_LOOPS=TEST_LOOPS+1))
echo -e "test loop $TEST_LOOPS          \\r";
done
echo "Found a failure:"
echo "$MYVAR" > run.log
echo "$MYVAR"
