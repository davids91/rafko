LIBFLAGS = `pkg-config --cflags protobuf grpc++`
LIBS = `pkg-config --libs protobuf grpc++`

DEBUG_FLAGS = -DNDEBUG #no debug
#DEBUG_FLAGS = -g #debug information added
#DEBUG_FLAGS = -ggdb #debug information added
#DEBUG_FLAGS += -v # verbose compile information
#DEBUG_FLAGS += -fsanitize=address #address sanitizer check
#DEBUG_FLAGS += -static-libasan #enables debug data to be shown in valgrind and gdb with addresssanitizer
CPPFLAGS = -Wall $(LIBFLAGS) $(LIBS) -std=c++14 $(DEBUG_FLAGS)
CPPFLAGS_DEPLOY = $(CPPFLAGS) -L$(LIB_DIR) -lsparsenetlib

##########################################################################################################
# GENERATED Objects
##########################################################################################################
CXX_GEN_FILES_DIR = ../gen

GENERATED_SOURCES = $(CXX_GEN_FILES_DIR)/common.pb.cc
GENERATED_SOURCES += $(CXX_GEN_FILES_DIR)/sparse_net.pb.cc $(CXX_GEN_FILES_DIR)/solution.pb.cc $(CXX_GEN_FILES_DIR)/training.pb.cc
GENERATED_SOURCES += $(CXX_GEN_FILES_DIR)/deep_learning_service.pb.cc $(CXX_GEN_FILES_DIR)/deep_learning_service.grpc.pb.cc
GENERATED_SOURCES += $(CXX_GEN_FILES_DIR)/logger.pb.cc

GENERATED_OBJECTS = $(subst $(CXX_GEN_FILES_DIR)/,,$(GENERATED_SOURCES:.cc=.o))

##########################################################################################################
# MAINFRAME
##########################################################################################################
MAINFRAME_SOURCE_DIR = rafko_mainframe

MAINFRAME_SOURCE_FILES = /services/src/deep_learning_server.cc
MAINFRAME_SOURCE_FILES += /services/src/server_slot.cc
MAINFRAME_SOURCE_FILES += /services/src/server_slot_run_net.cc
MAINFRAME_SOURCE_FILES += /services/src/server_slot_approximize_net.cc
MAINFRAME_SOURCE_FILES += /services/src/training_logger.cc
MAINFRAME_SOURCE_FILES += /control/src/rafko_deep_learning_mainframe.cc

MAINFRAME_SOURCES_TMP2 = $(subst /services, ../$(MAINFRAME_SOURCE_DIR)/services,$(MAINFRAME_SOURCE_FILES))
MAINFRAME_SOURCES_TMP = $(subst /models, ../$(MAINFRAME_SOURCE_DIR)/models,$(MAINFRAME_SOURCES_TMP2))
MAINFRAME_SOURCES = $(subst /control, ../$(MAINFRAME_SOURCE_DIR)/control,$(MAINFRAME_SOURCES_TMP2))
MAINFRAME_OBJECTS_TMP = $(subst ../$(MAINFRAME_SOURCE_DIR)/models/src/,,$(subst ../$(MAINFRAME_SOURCE_DIR)/services/src/,,$(MAINFRAME_SOURCES:.cc=.o)))
MAINFRAME_OBJECTS = $(subst /control/src, $(SPARSE_NET_SOURCE_DIR)/control/src,$(MAINFRAME_OBJECTS_TMP))
MAINFRAME_OUTPUT = rafko-mainframe.out

##########################################################################################################
# UTILITIES
##########################################################################################################
UTILITIES_SOURCE_DIR = rafko_utilities
#UTILITIES_SOURCES = None! Only header files as of yet

##########################################################################################################
# SPARSE NETWORK LIBRARY
##########################################################################################################
SPARSE_NET_SOURCE_DIR = sparse_net_library

SPARSE_NET_SOURCES = /services/src/sparse_net_builder.cc /services/src/neuron_router.cc
SPARSE_NET_SOURCES += /services/src/solution_builder.cc /services/src/partial_solution_builder.cc
SPARSE_NET_SOURCES += /services/src/partial_solution_solver.cc /services/src/solution_solver.cc
SPARSE_NET_SOURCES += /models/src/dense_net_weight_initializer.cc /services/src/weight_updater.cc
SPARSE_NET_SOURCES += /services/src/backpropagation_queue_wrapper.cc
SPARSE_NET_SOURCES +=  /models/src/neuron_info.cc /models/src/transfer_function.cc
SPARSE_NET_SOURCES +=  /models/src/cost_function.cc
SPARSE_NET_SOURCES += /services/src/weight_experience_space.cc

LIBRARY_NET_SOURCES_TMP = $(subst /services, ../$(SPARSE_NET_SOURCE_DIR)/services,$(SPARSE_NET_SOURCES))
LIBRARY_NET_SOURCES = $(subst /models, ../$(SPARSE_NET_SOURCE_DIR)/models,$(LIBRARY_NET_SOURCES_TMP))
LIBRARY_NET_OBJECTS = $(subst ../$(SPARSE_NET_SOURCE_DIR)/models/src/,,$(subst ../$(SPARSE_NET_SOURCE_DIR)/services/src/,,$(LIBRARY_NET_SOURCES:.cc=.o)))

##########################################################################################################
# GYM
##########################################################################################################
GYM_SOURCE_DIR = rafko_gym

GYM_SOURCES = /services/src/sparse_net_approximizer.cc
GYM_SOURCES += /services/src/agent.cc
GYM_SOURCES += /models/src/data_aggregate.cc

LIBRARY_GYM_SOURCES_TMP = $(subst /services, ../$(GYM_SOURCE_DIR)/services,$(GYM_SOURCES))
LIBRARY_GYM_SOURCES = $(subst /models, ../$(GYM_SOURCE_DIR)/models,$(LIBRARY_GYM_SOURCES_TMP))
LIBRARY_GYM_OBJECTS = $(subst ../$(GYM_SOURCE_DIR)/models/src/,,$(subst ../$(GYM_SOURCE_DIR)/services/src/,,$(LIBRARY_GYM_SOURCES:.cc=.o)))

##########################################################################################################
# OUTPUT LIBRARY
##########################################################################################################
LIB_DIR = ../lib

LIB_OUTPUT = $(LIB_DIR)/libsparsenetlib.a
GENERATED_FILES_INCLUDE_DIR = -I $(CXX_GEN_FILES_DIR)
SOURCES_INCLUDE_DIR = -I ../
LIBRARY_OBJECTS = $(LIBRARY_NET_OBJECTS) $(LIBRARY_GYM_OBJECTS) $(GENERATED_OBJECTS)
LIBRARY_SOURCES = $(LIBRARY_NET_SOURCES) $(LIBRARY_GYM_SOURCES) $(GENERATED_SOURCES)

.PHONY: lib clean gen
#default maketarget: Sparse net library
$(LIB_OUTPUT): $(LIB_DIR) $(LIBRARY_OBJECTS)
	ar -r -v -s $(LIB_OUTPUT) $(LIBRARY_OBJECTS)

mainframe: $(LIB_OUTPUT)
	g++ $(MAINFRAME_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(MAINFRAME_OUTPUT)

$(LIB_DIR):
	@echo "Folder $(directory) does not exist yet!"
	@mkdir -p $@

$(LIBRARY_OBJECTS):
	g++ -c $(LIBRARY_SOURCES) $(SOURCES_INCLUDE_DIR) $(GENERATED_FILES_INCLUDE_DIR) $(CPPFLAGS)

clean:
	rm -f *.o *.exe *.out ../lib/libsparsenetlib.a ../lib/libsparsenetlib.so $(CXX_GEN_FILES_DIR)/*.cc $(CXX_GEN_FILES_DIR)/*.h


##########################################################################################################
# Protocol buffers
##########################################################################################################
PROTO_FILES_DIR = ../../proto

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
	protoc logger.proto --proto_path="$(PROTO_FILES_DIR)" --cpp_out="$(CXX_GEN_FILES_DIR)/" --grpc_out="$(CXX_GEN_FILES_DIR)" --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH)

##########################################################################################################
# Testing
##########################################################################################################
TEST_BASE_SOURCE = ../test/src/main_test.cc
SPARSE_NET_TEST_SOURCES = ../test/$(SPARSE_NET_SOURCE_DIR)/src/sparse_net_builder_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/solution_builder_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/partial_solution_solver_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/solution_solver_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/synapse_iterator_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/neuron_router_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/neuron_info_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/cost_function_mse_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/backprop_queue_wrapper_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/weight_updater_test.cc
SPARSE_NET_TEST_SOURCES += ../test/$(SPARSE_NET_SOURCE_DIR)/src/weight_experience_space_test.cc
UTILITIES_TEST_SOURCES += ../test/$(UTILITIES_SOURCE_DIR)/src/data_ringbuffer_test.cc
UTILITIES_TEST_SOURCES += ../test/$(UTILITIES_SOURCE_DIR)/src/data_pool_test.cc
UTILITIES_TEST_SOURCES += ../test/$(UTILITIES_SOURCE_DIR)/src/thread_group_test.cc
GYM_TEST_SOURCES = ../test/$(GYM_SOURCE_DIR)/src/data_aggregate_test.cc
OPTIMIZE_TEST_SOURCES = $(TEST_BASE_SOURCE) ../test/$(GYM_SOURCE_DIR)/src/approximizer_test.cc
TEST_SOURCES = $(TEST_BASE_SOURCE) $(SPARSE_NET_TEST_SOURCES) $(UTILITIES_TEST_SOURCES) $(GYM_TEST_SOURCE)
TEST_INCLUDES = -I ../test/
TEST_RESULT = test-results.out
TEST_LOOPS = 100

.PHONY: test build-tests run-tests test-solution-builder test-only loop-tests
test: clean gen $(LIB_OUTPUT) $(TEST_RESULT)
	make run-tests

test-only: $(LIB_OUTPUT) $(TEST_RESULT)
	make run-tests

$(TEST_RESULT):
	g++ $(TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)

run-tests: $(TEST_RESULT)
	./$< --use-colour yes -d yes

loop-tests:
	./loop-tests.sh

test-solution-builder: clean gen $(LIB_OUTPUT)
	@rm -f $(TEST_RESULT)
	g++ $(TEST_BASE) ../test/solution_builder_test.cc ../test/neuron_router_test.cc $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)
	./$(TEST_RESULT)

test-opt: clean gen $(LIB_OUTPUT) #as in optimization or approximization
	@rm -f $(TEST_RESULT)
	g++ $(OPTIMIZE_TEST_SOURCES) $(CPPFLAGS_DEPLOY) $(SOURCES_INCLUDE_DIR) -o $(TEST_RESULT)
	./$(TEST_RESULT)  --use-colour yes -a [approximize]

$(SPARSE_NET_TEST_OBJECTS):
	g++ -c $(UTILITIES_TEST_SOURCES) $(SPARSE_NET_TEST_SOURCES) $(SOURCES_INCLUDE_DIR)
