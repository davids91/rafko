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
#include "gen/common.pb.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/models/cost_function_mse.h"

#include <vector>
#include <memory>

namespace sparse_net_library_test {

using std::unique_ptr;

using sparse_net_library::Data_set;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Cost_function_mse;

/*###############################################################################################
 * Testing Data aggregate implementation and seeing if it converts @Data_set correctly
 * into the data item wih statistics, and take care of statistic error data correctly
 * */
TEST_CASE("Testing Data aggregate for non-seuqeuntial data", "[data-handling]" ) {
  uint32 sample_number = 50;
  sdouble32 expected_label = double_literal(50.0);
  sdouble32 set_distance = double_literal(10.0);

  /* Create @Data_set for non-sequential data and fill it with data */
  Data_set data_set = Data_set();
  data_set.set_input_size(1);
  data_set.set_feature_size(1);
  data_set.set_sequence_size(1);

  for(uint32 i = 0; i < sample_number; ++i){
    data_set.add_inputs(expected_label); /* Input should be irrelevant here */
    data_set.add_labels(expected_label);
  }

  /* Create @Data_aggregate from @Data_set */
  Data_aggregate data_agr(data_set, std::make_unique<Cost_function_mse>(1));

  /* Test statistics for it */
  CHECK(double_literal(1.0) == data_agr.get_error() ); /* Initial error should be exactly 1.0 */

  /* Set all features to the given distance */
  for(uint32 i = 0; i < sample_number; ++i){
    data_agr.set_feature_for_label(i,{expected_label - set_distance});
  }

  CHECK( /* Error: (distance^2)/2 */
    Approx(
      pow(set_distance,2)/double_literal(2.0)
    ).epsilon(0.00000000000001) == data_agr.get_error()
  );
}

} /* namespace sparse_net_library_test */
