/*! This file is part of davids91/Rafko.
 *
 *    Rafko is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Rafko is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#include "test/catch.hpp"

#include "sparse_net_global.h"
#include "models/cost_function.h"
#include "models/cost_function_mse.h"

#include <random>

namespace sparse_net_library_test {

using sparse_net_library::Cost_function_mse;

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
  sdouble32 distance = double_literal(10.0);

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

  /* one feature distance should be (double_literal(0.5) * (distance)^2 ) */
  Cost_function_mse cost(feature_size, dataset_size);
  /*CHECK(
    Approx(cost.get_error(dataset,featureset) / static_cast<sdouble32>(dataset_size)).epsilon(double_literal(0.00000000000001))
    == (double_literal(0.5) * pow(distance,2))
  ); issue #59 */
  for(uint16 sample_iterator=0; sample_iterator< dataset_size; ++sample_iterator){
    CHECK(
      Approx(cost.get_feature_error(dataset[sample_iterator], featureset[sample_iterator])).epsilon(double_literal(0.00000000000001))
      == (double_literal(0.5 * feature_size) * pow(distance,2)) / static_cast<sdouble32>(dataset_size)
    );
  }

  /* overall feature ditance should be  */
}

} /* namespace sparse_net_library_test */