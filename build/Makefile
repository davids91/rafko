# Makefile
LIBS = `pkg-config --cflags --libs protobuf` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

CPPFLAGS = -Wall $(LIBS) -std=c++11 #-v
CPPFLAGS_TEST = -L../lib/ -lsparsenetlib

NET_SOURCES = ../models/sNet.pb.cc ../models/transfer_function_info.cc ../models/weight_initializer.cc
NET_OBJECTS = sNet.pb.o transfer_function_info.o weight_initializer.o

BUILDER_SOURCES = ../services/sparse_net_builder.cc ../models/dense_net_weight_initializer.cc
BUILDER_OBJECTS = sparse_net_builder.o Dense_net_weight_initializer.o

SOLVER_SOURCES = ../services/sparse_net_solver.cc
SOLVER_OBJECTS = sparse_net_solver.o

LIBRARY_OBJECTS = $(NET_OBJECTS) $(BUILDER_OBJECTS) $(SOLVER_OBJECTS)

MODELS_INCLUDE_DIR = ../models/
SERVICES_INCLUDE_DIR = ../services/
SOURCES_INCLUDE_DIR = ..

LIB_OUTPUT = ../lib/libsparsenetlib.a

.PHONY: build_library clean
build_library: $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

$(NET_OBJECTS):
	@clear
	g++ -c $(NET_SOURCES) -I $(SOURCES_INCLUDE_DIR) -I $(MODELS_INCLUDE_DIR) $(CPPFLAGS)

$(BUILDER_OBJECTS):
	g++ -c $(BUILDER_SOURCES) -I $(SOURCES_INCLUDE_DIR) -I $(MODELS_INCLUDE_DIR) -I $(SERVICES_INCLUDE_DIR) $(CPPFLAGS)

$(SOLVER_OBJECTS):
	g++ -c $(SOLVER_SOURCES) -I $(SOURCES_INCLUDE_DIR) -I $(MODELS_INCLUDE_DIR) -I $(SERVICES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so


##########################################################################################################
# Testing
##########################################################################################################

TEST_SOURCES = ../test/main_test.cc ../test/builder_test.cc ../test/solver_test.cc
TEST_OBJECTS = main_test.o builder_test.o solver_test.o
TEST_INCLUDES = ../test/
TEST_RESULT = test-result.out

.PHONY: test build-tests
build-tests: clean build_library $(TEST_OBJECTS) $(LIBRARY_OBJECTS)
	g++ $(TEST_OBJECTS)  $(CPPFLAGS_TEST) -I $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: build-tests
	./$(TEST_RESULT) --use-colour yes

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) -I $(SOURCES_INCLUDE_DIR)
