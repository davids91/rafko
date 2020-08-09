# Makefile
LIBFLAGS = `pkg-config --cflags protobuf grpc++`
LIBS = `pkg-config --libs protobuf grpc++`

LIB_DIR = ../lib
LIB_OUTPUT = $(LIB_DIR)/libsparsenetlib.a
MAINFRAME_OUTPUT = rafko-mainframe

DEBUG_FLAGS = -DNDEBUG #no debug
#DEBUG_FLAGS = -g #debug information added
#DEBUG_FLAGS = -ggdb #debug information added
#DEBUG_FLAGS += -v # verbose compile information
#DEBUG_FLAGS += -fsanitize=address #address sanitizer check
#DEBUG_FLAGS += -static-libasan #enables debug data to be shown in valgrind and gdb with addresssanitizer
CPPFLAGS = -Wall $(LIBFLAGS) $(LIBS) -std=c++14 $(DEBUG_FLAGS)
CPPFLAGS_DEPLOY = $(CPPFLAGS) -L$(LIB_DIR) -lsparsenetlib

PROTO_FILES_DIR = ../../proto
CXX_GEN_FILES_DIR = ../gen

##########################################################################################################
# Source files
##########################################################################################################
GENERATED_SOURCES = $(CXX_GEN_FILES_DIR)/common.pb.cc
GENERATED_SOURCES += $(CXX_GEN_FILES_DIR)/sparse_net.pb.cc $(CXX_GEN_FILES_DIR)/solution.pb.cc $(CXX_GEN_FILES_DIR)/training.pb.cc
GENERATED_SOURCES += $(CXX_GEN_FILES_DIR)/deep_learning_service.pb.cc $(CXX_GEN_FILES_DIR)/deep_learning_service.grpc.pb.cc

BUILDER_SOURCES = ../sparse_net_library/services/src/sparse_net_builder.cc ../sparse_net_library/services/src/neuron_router.cc
BUILDER_SOURCES += ../sparse_net_library/services/src/solution_builder.cc ../sparse_net_library/services/src/partial_solution_builder.cc
SOLVER_SOURCES = ../sparse_net_library/services/src/partial_solution_solver.cc ../sparse_net_library/services/src/solution_solver.cc
TRAINER_SOURCES = ../sparse_net_library/models/src/dense_net_weight_initializer.cc ../sparse_net_library/services/src/weight_updater.cc
TRAINER_SOURCES += ../sparse_net_library/services/src/backpropagation_queue_wrapper.cc
TRAINER_SOURCES += ../sparse_net_library/services/src/sparse_net_optimizer.cc ../sparse_net_library/services/src/sparse_net_approximizer.cc
HELPER_SOURCES =  ../sparse_net_library/models/src/neuron_info.cc ../sparse_net_library/models/src/transfer_function.cc
HELPER_SOURCES +=  ../sparse_net_library/models/src/cost_function.cc ../sparse_net_library/models/src/data_aggregate.cc

LIBRARY_SOURCES = $(GENERATED_SOURCES) $(BUILDER_SOURCES) $(SOLVER_SOURCES) $(TRAINER_SOURCES) $(HELPER_SOURCES)
LIBRARY_OBJECTS = $(subst $(CXX_GEN_FILES_DIR)/,,$(GENERATED_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../sparse_net_library/services/src/,,$(BUILDER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../sparse_net_library/services/src/,,$(SOLVER_SOURCES:.cc=.o))
LIBRARY_OBJECTS += $(subst ../sparse_net_library/models/src/,,$(subst ../sparse_net_library/services/src/,,$(TRAINER_SOURCES:.cc=.o)))
LIBRARY_OBJECTS += $(subst ../sparse_net_library/models/src/,,$(subst ../sparse_net_library/services/src/,,$(HELPER_SOURCES:.cc=.o)))

###########################
SERVER_SOURCES = ../rafko_mainframe/services/src/deep_learning_server.cc
SERVER_SOURCES += ../rafko_mainframe/services/src/server_slot.cc
SERVER_SOURCES += ../rafko_mainframe/services/src/server_slot_run_net.cc
SERVER_SOURCES += ../rafko_mainframe/services/src/server_slot_approximize_net.cc

MAINFRAME_SOURCES = $(SERVER_SOURCES)
MAINFRAME_SOURCES += ../rafko_mainframe/src/rafko_deep_learning_mainframe.cc

MAINFRAME_OBJECTS = $(subst ../rafko_mainframe/services/src/,,$(subst ../,,$(MAINFRAME_SOURCES:.cc=.o))))

GENERATED_FILES_INCLUDE_DIR = -I $(CXX_GEN_FILES_DIR)
SOURCES_INCLUDE_DIR = -I ../

.PHONY: lib clean gen
build_mainframe: $(LIB_OUTPUT) $(GENERATED_SOURCES)
	g++ $(GENERATED_SOURCES) $(MAINFRAME_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(MAINFRAME_OUTPUT)

$(LIB_OUTPUT): $(LIB_DIR) clean gen $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

##########################################################################################################
# Protocol bufffer
##########################################################################################################
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

$(GENERATED_SOURCES):
	make gen

run: $(MAINFRAME_OUTPUT)
	@ip addr | grep 192.
	./$(MAINFRAME_OUTPUT)

gen: #as in generate models; Can't be bothered to type something like `make generate-protocol-buffer-files` every time...
	protoc common.proto --proto_path="$(PROTO_FILES_DIR)" --cpp_out="$(CXX_GEN_FILES_DIR)/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc sparse_net.proto --proto_path="$(PROTO_FILES_DIR)" --cpp_out="$(CXX_GEN_FILES_DIR)/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc solution.proto --proto_path="$(PROTO_FILES_DIR)" --cpp_out="$(CXX_GEN_FILES_DIR)/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc training.proto --proto_path="$(PROTO_FILES_DIR)" --cpp_out="$(CXX_GEN_FILES_DIR)/" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)
	protoc deep_learning_service.proto --proto_path="$(PROTO_FILES_DIR)" --cpp_out="$(CXX_GEN_FILES_DIR)/" --grpc_out="$(CXX_GEN_FILES_DIR)" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)

##########################################################################################################
# Compile
##########################################################################################################
$(LIB_DIR):
	@echo "Folder $(directory) does not exist yet!"
	@mkdir -p $@

$(LIBRARY_OBJECTS):
	g++ -c $(LIBRARY_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so $(CXX_GEN_FILES_DIR)/*.cc $(CXX_GEN_FILES_DIR)/*.h

##########################################################################################################
# Testing
##########################################################################################################
TEST_BASE_SOURCE = ../test/src/main_test.cc
TEST_SOURCES = $(TEST_BASE_SOURCE)
TEST_SOURCES += ../test/src/sparse_net_builder_test.cc ../test/src/solution_builder_test.cc
TEST_SOURCES += ../test/src/partial_solution_solver_test.cc ../test/src/solution_solver_test.cc
TEST_SOURCES += ../test/src/synapse_iterator_test.cc ../test/src/neuron_router_test.cc
TEST_SOURCES += ../test/src/neuron_info_test.cc ../test/src/cost_function_mse_test.cc
TEST_SOURCES += ../test/src/backprop_queue_wrapper_test.cc
TEST_SOURCES += ../test/src/data_aggregate_test.cc
TEST_SOURCES += ../test/src/data_ringbuffer_test.cc
TEST_SOURCES += ../test/src/weight_updater_test.cc

OPTIMIZATION_TEST_SOURCES = $(TEST_BASE_SOURCE)
OPTIMIZATION_TEST_SOURCES += ../test/src/optimizer_test.cc ../test/src/approximizer_test.cc

TEST_OBJECTS = $(subst ../test/src/,,$(TEST_SOURCES:.cc=.o))
OPTIMIZATION_TEST_OBJECTS = $(subst ../test/src/,,$(OPTIMIZATION_TEST_SOURCES:.cc=.o))

TEST_INCLUDES = -I ../test/
TEST_RESULT = test-results.out
TEST_LOOPS = 100

.PHONY: test build-tests run-tests test-solution-builder test-only loop-tests
$(TEST_RESULT): $(TEST_OBJECTS)
	g++ $(TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)

test: clean gen $(LIB_OUTPUT)
	make run-tests

test-only: $(TEST_OBJECTS)
	g++ $(TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)
	make run-tests

run-tests: $(TEST_RESULT)
	./$< --use-colour yes

loop-tests:
	./loop-tests.sh

test-solution-builder: clean gen $(LIB_OUTPUT)
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) ../test/solution_builder_test.cc ../test/neuron_router_test.cc $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)
	./$(TEST_RESULT)

test-opt: clean gen $(LIB_OUTPUT) #as in optimization or approximization
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) $(OPTIMIZATION_TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)
	./$(TEST_RESULT)  --use-colour yes

$(TEST_OBJECTS):
	g++ -c $(TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
