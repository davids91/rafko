#include "test/catch.hpp"

#include "sparse_net_global.h"

#include "models/cost_function_quadratic.h"

#include <random>

namespace sparse_net_library_test {

using sparse_net_library::uint16;
using sparse_net_library::uint32;
using sparse_net_library::sdouble32;
using sparse_net_library::Cost_function_quadratic;

/*###############################################################################################
 * Testing Error function
 * - Create a dummy featureset and labelset with a given distance
 * - Calculate the distances to it
 * */
TEST_CASE( "Error function test", "[training][error-function]" ) {

  using std::vector;

  /* create fake data and fake features with a given distance */
  uint16 dataset_size = 500;
  uint32 feature_size = 20;
  sdouble32 distance = 10.0;

  vector<vector<sdouble32>> dataset = vector<vector<sdouble32>>(dataset_size,vector<sdouble32>(feature_size));
  vector<vector<sdouble32>> featureset = vector<vector<sdouble32>>(dataset_size,vector<sdouble32>(feature_size));
  srand(time(nullptr));
  for(uint16 sample_iterator=0; sample_iterator< dataset_size; ++sample_iterator){
      for(uint16 feature_iterator=0; feature_iterator< feature_size; ++feature_iterator){
        dataset[sample_iterator][feature_iterator] = static_cast<sdouble32>(rand()%dataset_size);
        if(0 == (rand()%2)) /* For every data feature and every sample, the distance should be equal to the correspoding datapoints */
          featureset[sample_iterator][feature_iterator] = dataset[sample_iterator][feature_iterator] + distance;
        else featureset[sample_iterator][feature_iterator] = dataset[sample_iterator][feature_iterator] - distance;
      }
  }

  /* one feature distance should be (0.5 * (distance)^2 ) */
  Cost_function_quadratic cost = Cost_function_quadratic(dataset);
  for(uint16 feature_iterator=0; feature_iterator< feature_size; ++feature_iterator){
    CHECK( cost.get_error(feature_iterator, featureset) ==  (0.5 * pow(distance,2)) );
  }

  /* overall feature ditance should be  */
}

} /* namespace sparse_net_library_test */
