
#include "test/catch.hpp"

#include "models/gen/sparse_net.pb.h"
#include "services/sparse_net_builder.h"

#include "models/gen/solution.pb.h"
#include "services/solution_builder.h"

namespace sparse_net_library_test {

using std::unique_ptr;
using std::make_unique;

using sparse_net_library::SparseNetBuilder;
using sparse_net_library::SparseNet;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution;

/*###############################################################################################
 * Testing Solution generation using the @Sparse_net_builder and the @Solution_builder
 * */
Solution* test_solution_builder_manually(google::protobuf::Arena* arena){
  /* Build a net */
  unique_ptr<SparseNetBuilder> net_builder = make_unique<SparseNetBuilder>();
  net_builder->input_size(5)
    .input_neuron_size(2).output_neuron_number(2).expectedInputRange(5.0)
    .arena_ptr(arena);
  SparseNet* net(net_builder->denseLayers({2,3,2}));
  net_builder.reset();

  /* Generate solution from Net */
  unique_ptr<Solution_builder> solution_builder = make_unique<Solution_builder>();
  solution_builder->max_solve_threads(1).device_max_megabytes(100.0)
    .arena_ptr(arena);

  return solution_builder->build(*net);
}

TEST_CASE( "Building a solution from a net", "[build][small]" ){
  Solution* solution = test_solution_builder_manually(nullptr);
  REQUIRE( nullptr != solution );
  delete solution;
}

} /* namespace sparse_net_library_test */