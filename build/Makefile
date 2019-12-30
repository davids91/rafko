# Makefile
LIBS = `pkg-config --cflags --libs protobuf` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

CPPFLAGS = -Wall $(LIBS) -std=c++14 -DNDEBUG -g #-v
CPPFLAGS_TEST = -Wall -std=c++14 -L../lib/ -lsparsenetlib -g

NET_SOURCES = ../cxx/gen/common.pb.cc ../cxx/gen/sparse_net.pb.cc ../cxx/gen/solution.pb.cc
NET_OBJECTS = common.pb.o sparse_net.pb.o solution.pb.o 

BUILDER_SOURCES = ../cxx/services/src/sparse_net_builder.cc 
BUILDER_SOURCES += ../cxx/services/src/solution_builder.cc ../cxx/services/src/partial_solution_builder.cc
BUILDER_OBJECTS = sparse_net_builder.o solution_builder.o partial_solution_builder.o

SOLVER_SOURCES = ../cxx/services/src/sparse_net_solver.cc
SOLVER_SOURCES += ../cxx/services/src/partial_solution_solver.cc ../cxx/services/src/solution_solver.cc
SOLVER_OBJECTS = sparse_net_solver.o partial_solution_solver.o solution_solver.o

HELPER_SOURCES = ../cxx/models/src/transfer_function_info.cc ../cxx/services/src/synapse_iterator.cc
HELPER_SOURCES += ../cxx/models/src/weight_initializer.cc ../cxx/models/src/dense_net_weight_initializer.cc
HELPER_SOURCES += ../cxx/services/src/neuron_router.cc ../cxx/models/src/neuron_info.cc
HELPER_OBJECTS = transfer_function_info.o synapse_iterator.o
HELPER_OBJECTS += weight_initializer.o dense_net_weight_initializer.o
HELPER_OBJECTS += neuron_router.o  neuron_info.o

LIBRARY_OBJECTS = $(NET_OBJECTS) $(BUILDER_OBJECTS) $(SOLVER_OBJECTS) $(HELPER_OBJECTS)

GENERATED_FILES_INCLUDE_DIR = -I ../cxx/gen
SOURCES_INCLUDE_DIR = -I ../cxx

LIB_OUTPUT = ../lib/libsparsenetlib.a

.PHONY: build_library clean gen-model
build_library: clean $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

gen: #as in generate models; Can't be bothered to type something like `make generate-protocol-buffer-files` every time../cxx.
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

TEST_BASE = ../cxx/test/src/main_test.cc
TEST_SOURCES = $(TEST_BASE) ../cxx/test/src/net_builder_test.cc ../cxx/test/src/solution_builder_test.cc
TEST_SOURCES += ../cxx/test/src/partial_solution_solver_test.cc ../cxx/test/src/solution_solver_test.cc
TEST_SOURCES += ../cxx/test/src/synapse_iterator_test.cc ../cxx/test/src/neuron_router_test.cc
TEST_SOURCES += ../cxx/test/src/neuron_info_test.cc
TEST_OBJECTS = main_test.o net_builder_test.o solution_builder_test.o
TEST_OBJECTS += partial_solution_solver_test.o solution_solver_test.o
TEST_OBJECTS += synapse_iterator_test.o neuron_router_test.o
TEST_OBJECTS += neuron_info_test.o
TEST_INCLUDES = -I ../cxx/test/
TEST_RESULT = test-results.out
TEST_LOOPS = 100

.PHONY: test build-tests run-tests test-solution-builder test-only loop-tests
build-tests: $(TEST_OBJECTS) $(LIBRARY_OBJECTS)
	@rm -f $(TEST_RESULT)
	g++ $(TEST_SOURCES) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: clean build_library build-tests
	./$(TEST_RESULT) --use-colour yes

test-only: $(TEST_OBJECTS)
	@clear
	g++ $(TEST_SOURCES) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	make run-tests

run-tests: $(TEST_RESULT)
	./$(TEST_RESULT) --use-colour yes

loop-tests:
	for ((i=1; i <= ${TEST_LOOPS}; ++i)) do ./$(TEST_RESULT) --use-colour yes | grep FAILED: -A 5; done

test-solution-builder: clean build_library
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) ../cxx/test/solution_builder_test.cc ../cxx/test/neuron_router_test.cc $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	make run-tests

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
