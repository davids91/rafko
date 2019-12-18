# Makefile
LIBS = `pkg-config --cflags --libs protobuf` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

CPPFLAGS = -Wall $(LIBS) -std=c++14 #-v
CPPFLAGS_TEST = -L../lib/ -lsparsenetlib

NET_SOURCES = ../models/src/gen/sparse_net.pb.cc ../models/src/gen/solution.pb.cc ../models/src/transfer_function_info.cc
NET_OBJECTS = sparse_net.pb.o solution.pb.o transfer_function_info.o

BUILDER_SOURCES = ../services/src/sparse_net_builder.cc ../services/src/solution_builder.cc ../models/src/weight_initializer.cc ../models/src/dense_net_weight_initializer.cc
BUILDER_OBJECTS = sparse_net_builder.o weight_initializer.o Dense_net_weight_initializer.o

SOLVER_SOURCES = ../services/src/sparse_net_solver.cc ../services/src/partial_solution_solver.cc
SOLVER_OBJECTS = sparse_net_solver.o partial_solution_solver.o

LIBRARY_OBJECTS = $(NET_OBJECTS) $(BUILDER_OBJECTS) $(SOLVER_OBJECTS)

GENERATED_FILES_INCLUDE_DIR = -I ../models/gen
SOURCES_INCLUDE_DIR = -I .. 

LIB_OUTPUT = ../lib/libsparsenetlib.a

.PHONY: build_library clean
build_library: clean $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

$(NET_OBJECTS):
	@clear
	g++ -c $(NET_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

$(BUILDER_OBJECTS):
	g++ -c $(BUILDER_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

$(SOLVER_OBJECTS):
	g++ -c $(SOLVER_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so


##########################################################################################################
# Testing
##########################################################################################################

TEST_SOURCES = ../test/main_test.cc ../test/net_builder_test.cc ../test/partial_solution_solver_test.cc
TEST_OBJECTS = main_test.o net_builder_test.o partial_solution_solver_test.o
TEST_INCLUDES = -I ../test/
TEST_RESULT = test-result.out

.PHONY: test build-tests run-tests
build-tests: $(TEST_OBJECTS) $(LIBRARY_OBJECTS)
	@rm -f test-result.out
	g++ $(TEST_OBJECTS) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: clean build_library build-tests
	./$(TEST_RESULT) --use-colour yes

run-tests: $(TEST_RESULT)
	./$(TEST_RESULT) --use-colour yes

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
