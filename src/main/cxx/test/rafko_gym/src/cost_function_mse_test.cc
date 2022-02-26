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
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/services/cost_function.h"
#include "rafko_gym/services/cost_function_mse.h"

#include "test/test_utility.h"

namespace rafko_net_test {

/*###############################################################################################
 * Testing Error function
 * - Create a dummy featureset and labelset with a given distance
 * - Calculate the distances to it
 * */
TEST_CASE( "Error function mean squared error test", "[training][error-function]" ) {
  rafko_mainframe::RafkoSettings settings;

  /* create fake data and fake features with a given distance */
  std::uint16_t dataset_size = 500;
  std::uint32_t feature_size = 20;
  double distance = (10.0);

  std::vector<std::vector<double>> dataset = std::vector<std::vector<double>>(dataset_size,std::vector<double>(feature_size));
  std::vector<std::vector<double>> featureset = std::vector<std::vector<double>>(dataset_size,std::vector<double>(feature_size));
  for(std::uint16_t sample_iterator=0; sample_iterator< dataset_size; ++sample_iterator){
      for(std::uint16_t feature_iterator=0; feature_iterator< feature_size; ++feature_iterator){
        dataset[sample_iterator][feature_iterator] = static_cast<double>(rand()%dataset_size);
        if(0 == (rand()%2)) /* For every data feature and every sample, the distance should be equal to the correspoding datapoints */
          featureset[sample_iterator][feature_iterator] = dataset[sample_iterator][feature_iterator] + distance;
        else featureset[sample_iterator][feature_iterator] = dataset[sample_iterator][feature_iterator] - distance;
      }
  }

  /* one feature distance should be ((0.5) * (distance)^2 ) */
  rafko_gym::CostFunctionMSE cost(settings);
  for(std::uint16_t sample_iterator = 0; sample_iterator < dataset_size; ++sample_iterator){
    REQUIRE(
      Catch::Approx(
        cost.get_feature_error(dataset[sample_iterator], featureset[sample_iterator], dataset_size)
      ).epsilon((0.00000000000001))
      == ((0.5) * feature_size * pow(distance,2)) / static_cast<double>(dataset_size)
    );
  }

  /* Test if the whole dataset can be processed in one function call */
  std::vector<double> label_errors(dataset_size,0);
  cost.get_feature_errors(dataset, featureset, label_errors, 0, 0, label_errors.size(), 0, dataset_size);
  for(const double label_error : label_errors){
    CHECK(
      Catch::Approx(label_error).epsilon((0.00000000000001))
      == ((0.5) * feature_size * pow(distance,2)) / static_cast<double>(dataset_size)
    );
  }

}

} /* namespace rafko_net_test */
