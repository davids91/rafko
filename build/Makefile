# Makefile
LIBS = `pkg-config --cflags --libs protobuf grpc++` #-pthread -IC:/msys64/mingw64/include -LC:/msys64/mingw64/lib -lprotobuf
LIB_DIR = ../lib
LIB_OUTPUT = $(LIB_DIR)/libsparsenetlib.a

DEBUG_FLAGS = -DNDEBUG #no debug
#DEBUG_FLAGS = -g #debug information added
#DEBUG_FLAGS = -ggdb #debug information added
#DEBUG_FLAGS += -v # verbose compile information
#DEBUG_FLAGS += -fsanitize=address #address sanitizer check
#DEBUG_FLAGS += -static-libasan #enables debug data to be shown in valgrind and gdb with addresssanitizer
CPPFLAGS = -Wall $(LIBS) -std=c++14 $(DEBUG_FLAGS)
CPPFLAGS_DEPLOY = $(CPPFLAGS) -L$(LIB_DIR) -lsparsenetlib

##########################################################################################################
# Source files
##########################################################################################################
GENERATED_SOURCES = ../cxx/gen/common.pb.cc 
GENERATED_SOURCES += ../cxx/gen/sparse_net.pb.cc ../cxx/gen/solution.pb.cc ../cxx/gen/training.pb.cc
GENERATED_SOURCES += ../cxx/gen/deep_learning_service.pb.cc ../cxx/gen/deep_learning_service.grpc.pb.cc

BUILDER_SOURCES = ../cxx/services/src/sparse_net_builder.cc ../cxx/services/src/neuron_router.cc
BUILDER_SOURCES += ../cxx/services/src/solution_builder.cc ../cxx/services/src/partial_solution_builder.cc
SOLVER_SOURCES = ../cxx/services/src/partial_solution_solver.cc ../cxx/services/src/solution_solver.cc
TRAINER_SOURCES = ../cxx/models/src/dense_net_weight_initializer.cc ../cxx/services/src/weight_updater.cc
TRAINER_SOURCES += ../cxx/services/src/backpropagation_queue_wrapper.cc
TRAINER_SOURCES += ../cxx/services/src/sparse_net_optimizer.cc ../cxx/services/src/sparse_net_approximizer.cc
HELPER_SOURCES =  ../cxx/models/src/neuron_info.cc ../cxx/models/src/transfer_function.cc

SERVER_SOURCES = ../cxx/services/src/deep_learning_server.cc

MAINFRAME_SOURCES = ../cxx/rafko_deep_learning_mainframe.cc

LIBRARY_SOURCES = $(GENERATED_SOURCES) $(BUILDER_SOURCES) $(SOLVER_SOURCES) $(TRAINER_SOURCES) $(SERVER_SOURCES) $(HELPER_SOURCES)
LIBRARY_OBJECTS = $(subst ../cxx/gen/,,$(GENERATED_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/services/src/,,$(BUILDER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/services/src/,,$(SOLVER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/services/src/,,$(SERVER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../cxx/models/src/,,$(subst ../cxx/services/src/,,$(TRAINER_SOURCES:.cc=.o)))
LIBRARY_OBJECTS += $(subst ../cxx/models/src/,,$(subst ../cxx/services/src/,,$(HELPER_SOURCES:.cc=.o)))

MAINFRAME_OBJECTS = $(subst ../cxx/,,$(MAINFRAME_SOURCES:.cc=.o))

GENERATED_FILES_INCLUDE_DIR = -I ../cxx/gen
SOURCES_INCLUDE_DIR = -I ../cxx

.PHONY: build_library clean gen
build_mainframe: $(LIB_OUTPUT)
	g++ $(GENERATED_SOURCES) $(SERVER_SOURCES) $(MAINFRAME_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o rafko_mainframe

build_library: clean gen $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

##########################################################################################################
# Protocol bufffer
##########################################################################################################
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

gen: #as in generate models; Can't be bothered to type something like `make generate-protocol-buffer-files` every time...
	protoc common.proto --proto_path="../proto/" --cpp_out="../cxx/gen/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc sparse_net.proto --proto_path="../proto/" --cpp_out="../cxx/gen/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc solution.proto --proto_path="../proto/" --cpp_out="../cxx/gen/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc training.proto --proto_path="../proto/" --cpp_out="../cxx/gen/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc deep_learning_service.proto --proto_path="../proto/" --cpp_out="../cxx/gen/" --grpc_out="../cxx/gen" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)

##########################################################################################################
# Compile
##########################################################################################################
$(LIB_DIR):
	@echo "Folder $(directory) does not exist"
	mkdir -p $@

$(LIBRARY_OBJECTS):
	g++ -c $(LIBRARY_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so ../cxx/gen/*.cc ../cxx/gen/*.h

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
TEST_SOURCES += ../cxx/test/src/data_aggregate_test.cc
TEST_SOURCES += ../cxx/test/src/data_ringbuffer_test.cc
TEST_SOURCES += ../cxx/test/src/weight_updater_test.cc

OPTIMIZATION_TEST_SOURCES = $(TEST_BASE_SOURCE)
OPTIMIZATION_TEST_SOURCES += ../cxx/test/src/optimizer_test.cc ../cxx/test/src/approximizer_test.cc

TEST_OBJECTS = $(subst ../cxx/test/src/,,$(TEST_SOURCES:.cc=.o))
OPTIMIZATION_TEST_OBJECTS = $(subst ../cxx/test/src/,,$(OPTIMIZATION_TEST_SOURCES:.cc=.o))

TEST_INCLUDES = -I ../cxx/test/
TEST_RESULT = test-results.out
TEST_LOOPS = 100

.PHONY: test build-tests run-tests test-solution-builder test-only loop-tests
$(TEST_RESULT): $(TEST_OBJECTS)
	g++ $(TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)

test: clean build_library
	make run-tests

test-only: $(TEST_OBJECTS)
	g++ $(TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	make run-tests

run-tests: $(TEST_RESULT)
	./$< --use-colour yes

loop-tests:
	while true; do ./$(TEST_RESULT) --use-colour yes; done

test-solution-builder: clean build_library
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) ../cxx/test/solution_builder_test.cc ../cxx/test/neuron_router_test.cc $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	./$(TEST_RESULT)

test-opt: clean build_library #as in optimization or approximization
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) $(OPTIMIZATION_TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) $(CPPFLAGS) -o $(TEST_RESULT)
	./$(TEST_RESULT) [approximize]

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
