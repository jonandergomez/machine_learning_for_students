
#
# This makefile prepares the TGZ file to be used when invoking some
# examples in a Spark environment. See examples in the testing directory.
#

all: clean
	tar zcvf mypythonlib.tgz  ann machine_learning   testing


clean:
	find . -name "*.pyc" -exec rm {} \;
