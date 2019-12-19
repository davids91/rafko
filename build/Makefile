# Makefile
LIBS = `pkg-config --cflags --libs protobuf` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

CPPFLAGS = -Wall $(LIBS) -std=c++14 -DNDEBUG #-v
CPPFLAGS_TEST = -Wall -std=c++14 -L../lib/ -lsparsenetlib

NET_SOURCES = ../models/src/gen/sparse_net.pb.cc ../models/src/gen/solution.pb.cc
NET_OBJECTS = sparse_net.pb.o solution.pb.o

BUILDER_SOURCES = ../services/src/sparse_net_builder.cc ../services/src/solution_builder.cc
BUILDER_OBJECTS = sparse_net_builder.o solution_builder.o

SOLVER_SOURCES = ../services/src/sparse_net_solver.cc ../services/src/partial_solution_solver.cc
SOLVER_OBJECTS = sparse_net_solver.o partial_solution_solver.o

HELPER_SOURCES = ../models/src/transfer_function_info.cc ../services/src/synapse_iterator.cc
HELPER_SOURCES += ../models/src/weight_initializer.cc ../models/src/dense_net_weight_initializer.cc
HELPER_SOURCES +=  ../services/src/neuron_router.cc ../models/src/neuron_info.cc
HELPER_OBJECTS = transfer_function_info.o synapse_iterator.o neuron_router.o weight_initializer.o dense_net_weight_initializer.o neuron_info.o

LIBRARY_OBJECTS = $(NET_OBJECTS) $(BUILDER_OBJECTS) $(SOLVER_OBJECTS) $(HELPER_OBJECTS)

GENERATED_FILES_INCLUDE_DIR = -I ../models/gen
SOURCES_INCLUDE_DIR = -I ..

LIB_OUTPUT = ../lib/libsparsenetlib.a

.PHONY: build_library clean gen-model
build_library: clean $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

gen: #as in generate models; Can't be bothered to type something like `make generate-protocol-buffer-files` every time...
	./gen_models.bat

$(NET_OBJECTS):
	@clear
	g++ -c $(NET_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

$(BUILDER_OBJECTS):
	g++ -c $(BUILDER_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

$(SOLVER_OBJECTS):
	g++ -c $(SOLVER_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

$(HELPER_OBJECTS):
	g++ -c $(HELPER_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so


##########################################################################################################
# Testing
##########################################################################################################

TEST_BASE = ../test/main_test.cc
TEST_SOURCES = $(TEST_BASE) ../test/net_builder_test.cc ../test/partial_solution_solver_test.cc ../test/solution_builder_test.cc ../test/neuron_router_test.cc
TEST_OBJECTS = main_test.o net_builder_test.o partial_solution_solver_test.o solution_builder_test.o neuron_router_test.o
TEST_INCLUDES = -I ../test/
TEST_RESULT = test-result.out

.PHONY: test build-tests run-tests test-solution-builder
build-tests: $(TEST_OBJECTS) $(LIBRARY_OBJECTS)
	@rm -f $(TEST_RESULT)
	g++ $(TEST_SOURCES) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: clean build_library build-tests
	./$(TEST_RESULT) --use-colour yes

run-tests: $(TEST_RESULT)
	./$(TEST_RESULT) --use-colour yes

test-solution-builder: clean build_library
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) ../test/solution_builder_test.cc ../test/neuron_router_test.cc $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	make run-tests

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
