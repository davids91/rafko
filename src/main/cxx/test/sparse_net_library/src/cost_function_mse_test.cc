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
#include "test/test_utility.h"

#include <random>

#include "rafko_mainframe/models/service_context.h"
#include "rafko_net/models/cost_function.h"
#include "rafko_net/models/cost_function_mse.h"


namespace sparse_net_library_test {

using sparse_net_library::Cost_function_mse;
using rafko_mainframe::Service_context;

using std::vector;

/*###############################################################################################
 * Testing Error function
 * - Create a dummy featureset and labelset with a given distance
 * - Calculate the distances to it
 * */
TEST_CASE( "Error function test", "[training][error-function]" ) {
  Service_context service_context;

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
  Cost_function_mse cost(feature_size, service_context);
  for(uint16 sample_iterator=0; sample_iterator< dataset_size; ++sample_iterator){
    CHECK(
      Approx(
        cost.get_feature_error(dataset[sample_iterator], featureset[sample_iterator], dataset_size)
      ).epsilon(double_literal(0.00000000000001))
      == (double_literal(0.5) * feature_size * pow(distance,2)) / static_cast<sdouble32>(dataset_size)
    );
  }

  /* Test if the whole dataset can be processed in one function call */
  vector<sdouble32> label_errors(dataset_size,0);
  cost.get_feature_errors(dataset, featureset, label_errors, 0, 0, label_errors.size(), 0, dataset_size);
  for(const sdouble32 label_error : label_errors){
    CHECK(
      Approx(label_error).epsilon(double_literal(0.00000000000001))
      == (double_literal(0.5) * feature_size * pow(distance,2)) / static_cast<sdouble32>(dataset_size)
    );
  }

}

} /* namespace sparse_net_library_test */
