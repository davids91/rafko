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

#include <random>
#include <math.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/services/cost_function.h"
#include "rafko_gym/services/cost_function_cross_entropy.h"

#include "test/test_utility.h"

namespace rafko_net_test {

/*###############################################################################################
 * Testing Error function
 * - Create a dummy featureset and labelset with a given distance
 * - Calculate the distances to it
 * */
TEST_CASE( "Error function cross entropy test", "[training][error-function]" ) {
  rafko_mainframe::RafkoSettings settings;

  /* create fake data and fake features with a given distance */
  uint16 dataset_size = 500;
  uint32 feature_size = 20;

  std::vector<std::vector<sdouble32>> dataset = std::vector<std::vector<sdouble32>>(dataset_size,std::vector<sdouble32>(feature_size));
  std::vector<std::vector<sdouble32>> featureset = std::vector<std::vector<sdouble32>>(dataset_size,std::vector<sdouble32>(feature_size));
  for(uint16 sample_iterator = 0u; sample_iterator< dataset_size; ++sample_iterator){
      for(uint16 feature_iterator=0; feature_iterator< feature_size; ++feature_iterator){
        if(0 == (rand()%2)) /* Create random classification vectors */
          dataset[sample_iterator][feature_iterator] = double_literal(1.0);
        else dataset[sample_iterator][feature_iterator] = double_literal(0.0000000000000001);
        if(0 == (rand()%2)) /* Create random classification vectors */
          featureset[sample_iterator][feature_iterator] = double_literal(1.0);
        else featureset[sample_iterator][feature_iterator] = double_literal(0.0000000000000001); /* don't use zero, because log(0) = -infinity! */
      }
  }

  rafko_gym::CostFunctionCrossEntropy cost(settings);
  std::vector<sdouble32> calculated_errors;
  for(uint16 sample_iterator = 0; sample_iterator < dataset_size; ++sample_iterator){
    calculated_errors.push_back(double_literal(0.0));
    for(uint32 i = 0u; i < dataset[sample_iterator].size(); ++i){
      calculated_errors[sample_iterator] += dataset[sample_iterator][i] * std::log(featureset[sample_iterator][i]);
    }
    calculated_errors[sample_iterator] /= static_cast<sdouble32>(dataset_size);
    REQUIRE(
      Catch::Approx(
        cost.get_feature_error(dataset[sample_iterator], featureset[sample_iterator], dataset_size)
      ).epsilon(double_literal(0.00000000000001)) == calculated_errors[sample_iterator]
    );
  }

  std::vector<sdouble32> label_errors(dataset_size,0);
  cost.get_feature_errors(dataset, featureset, label_errors, 0, 0, label_errors.size(), 0, dataset_size);
  for(uint16 sample_iterator = 0; sample_iterator < dataset_size; ++sample_iterator){
    CHECK( Catch::Approx(label_errors[sample_iterator]).epsilon(double_literal(0.00000000000001)) == calculated_errors[sample_iterator] );
  }
}

} /* namespace rafko_net_test */
