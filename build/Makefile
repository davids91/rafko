# Makefile
LIBS = `pkg-config --cflags --libs protobuf` -g#-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

LIBS = `pkg-config --cflags --libs protobuf` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf

DEBUG_FLAGS = -DNDEBUG #no debug
#DEBUG_FLAGS = -g #debug information added
#DEBUG_FLAGS = -ggdb #debug information added
#DEBUG_FLAGS += -v # verbose compile information
#DEBUG_FLAGS += -fsanitize=address #address sanitizer check
#DEBUG_FLAGS += -static-libasan #enables debug data to be shown in valgrind and gdb with addresssanitizer
CPPFLAGS = -Wall $(LIBS) -std=c++14 $(DEBUG_FLAGS)
CPPFLAGS_TEST = -Wall -std=c++14 -L../lib/ -lsparsenetlib $(DEBUG_FLAGS)

GENERATED_SOURCES = ../cxx/gen/common.pb.cc ../cxx/gen/sparse_net.pb.cc ../cxx/gen/solution.pb.cc

BUILDER_SOURCES = ../cxx/services/src/sparse_net_builder.cc
BUILDER_SOURCES += ../cxx/services/src/solution_builder.cc ../cxx/services/src/partial_solution_builder.cc

SOLVER_SOURCES = ../cxx/services/src/partial_solution_solver.cc ../cxx/services/src/solution_solver.cc
SOLVER_SOURCES += ../cxx/services/src/sparse_net_optimizer.cc

HELPER_SOURCES = ../cxx/services/src/synapse_iterator.cc
HELPER_SOURCES += ../cxx/models/src/dense_net_weight_initializer.cc
HELPER_SOURCES += ../cxx/services/src/neuron_router.cc ../cxx/models/src/neuron_info.cc
HELPER_SOURCES += ../cxx/models/src/transfer_function.cc
HELPER_SOURCES += ../cxx/services/src/backpropagation_queue_wrapper.cc ../cxx/models/src/data_aggregate.cc
HELPER_SOURCES += ../cxx/services/src/weight_updater.cc

LIBRARY_SOURCES = $(GENERATED_SOURCES) $(BUILDER_SOURCES) $(SOLVER_SOURCES) $(HELPER_SOURCES)
LIBRARY_OBJECTS = $(subst ../cxx/gen/,,$(GENERATED_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/services/src/,,$(BUILDER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/services/src/,,$(SOLVER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/models/src/,,$(subst ../cxx/services/src/,,$(HELPER_SOURCES:.cc=.o)))

GENERATED_FILES_INCLUDE_DIR = -I ../cxx/gen
SOURCES_INCLUDE_DIR = -I ../cxx

LIB_OUTPUT = ../lib/libsparsenetlib.a

.PHONY: build_library clean gen
build_library: clean $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

gen: #as in generate models; Can't be bothered to type something like `make generate-protocol-buffer-files` every time...
	./gen_models.bat

$(LIBRARY_OBJECTS):
	g++ -c $(LIBRARY_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so


##########################################################################################################
# Testing
##########################################################################################################

TEST_BASE_SOURCE = ../cxx/test/src/main_test.cc
TEST_SOURCES = $(TEST_BASE_SOURCE)
TEST_SOURCES += ../cxx/test/src/net_builder_test.cc ../cxx/test/src/solution_builder_test.cc
TEST_SOURCES += ../cxx/test/src/partial_solution_solver_test.cc ../cxx/test/src/solution_solver_test.cc
TEST_SOURCES += ../cxx/test/src/synapse_iterator_test.cc ../cxx/test/src/neuron_router_test.cc
TEST_SOURCES += ../cxx/test/src/neuron_info_test.cc ../cxx/test/src/cost_function_mse_test.cc
TEST_SOURCES += ../cxx/test/src/backprop_queue_wrapper_test.cc

OPTIMIZATION_TEST_SOURCES = $(TEST_BASE_SOURCE)
OPTIMIZATION_TEST_SOURCES += ../cxx/test/src/optimizer_test.cc

TEST_OBJECTS = $(subst ../cxx/test/src/,,$(TEST_SOURCES:.cc=.o))
OPTIMIZATION_TEST_OBJECTS = $(subst ../cxx/test/src/,,$(OPTIMIZATION_TEST_SOURCES:.cc=.o))

TEST_INCLUDES = -I ../cxx/test/
TEST_RESULT = test-results.out
TEST_LOOPS = 100

.PHONY: test build-tests run-tests test-solution-builder test-only loop-tests
$(TEST_RESULT): $(TEST_OBJECTS)
	@rm -f $(TEST_RESULT)
	g++ $(TEST_SOURCES) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: clean build_library run-tests

test-only: $(TEST_OBJECTS)
	g++ $(TEST_SOURCES) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	make run-tests

run-tests: $(TEST_RESULT)
	./$< --use-colour yes

loop-tests:
	for ((i=1; i <= ${TEST_LOOPS}; ++i)) do ./$(TEST_RESULT) --use-colour yes | grep FAILED: -A 5; done

test-solution-builder: clean build_library
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) ../cxx/test/solution_builder_test.cc ../cxx/test/neuron_router_test.cc $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	./$(TEST_RESULT)

test-opt: clean build_library #as in optimization
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) $(OPTIMIZATION_TEST_SOURCES) $(CPPFLAGS_TEST) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	./$(TEST_RESULT)

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
