# Makefile
LIBS = `pkg-config --cflags --libs protobuf` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

CPPFLAGS = -Wall $(LIBS) -std=c++11 #-v
CPPFLAGS_TEST = -L../lib/ -lsparsenetlib

NET_SOURCES = ../models/sNet.pb.cc
NET_OBJECTS = sNet.pb.o

BUILDER_SOURCES = ../services/snetbuilder.cc ../models/dense_net_weight_initializer.cc ../models/transferfunctioninfo.cc ../models/weight_initializer.cc
BUILDER_OBJECTS = snetbuilder.o Dense_net_weight_initializer.o transferfunctioninfo.o weight_initializer.o

LIBRARY_OBJECTS = $(NET_OBJECTS) $(BUILDER_OBJECTS)

MODELS_INCLUDE_DIR = ../models/
SERVICES_INCLUDE_DIR = ../services/
SOURCES_INCLUDE_DIR = ..

LIB_OUTPUT = ../lib/libsparsenetlib.a

.PHONY: build_library clean
build_library: $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

$(NET_OBJECTS):
	clear
	g++ -c $(NET_SOURCES) -I $(SOURCES_INCLUDE_DIR) -I $(MODELS_INCLUDE_DIR) $(CPPFLAGS)

$(BUILDER_OBJECTS):
	g++ -c $(BUILDER_SOURCES) -I $(SOURCES_INCLUDE_DIR) -I $(MODELS_INCLUDE_DIR) -I $(SERVICES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so


##########################################################################################################
# Testing
##########################################################################################################

TEST_SOURCES = ../test/builder_test.cc ../test/main_test.cc
TEST_OBJECTS = builder_test.o main_test.o
TEST_INCLUDES = ../test/
TEST_RESULT = test-result.out

.PHONY: test build-tests
build-tests: clean build_library $(TEST_OBJECTS) $(LIBRARY_OBJECTS)
	g++ $(TEST_OBJECTS)  $(CPPFLAGS_TEST) -I $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: build-tests
	./$(TEST_RESULT) --use-colour yes

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) -I $(SOURCES_INCLUDE_DIR)
