// Regression coverage for extract_sub / replace_sub on a non-contiguous
// Cytnx backend layout (the kind produced by tci::transpose via
// cytnx::Tensor::permute_). Cytnx's bulk get / set are mapper-aware, but
// the existing conformance suite only exercises the contiguous path.

#include <doctest/doctest.h>

#include <cytnx.hpp>

#include "tci/tci.h"

namespace {

  using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;
  using Elem = tci::elem_t<Tensor>;
  using ElemCoor = tci::elem_coor_t<Tensor>;
  using ElemCoors = tci::elem_coors_t<Tensor>;

  Elem expected_M(std::size_t i, std::size_t j) { return static_cast<Elem>(i * 10 + j); }

}  // namespace

TEST_CASE("extract_sub on transposed (non-contiguous) tensor") {
  tci::context_handle_t<Tensor> ctx;
  tci::create_context(ctx);

  // Build a 4x5 source tensor with unique per-coordinate values, then
  // transpose so the Cytnx-side _mapper is no longer the identity.
  Tensor M;
  tci::zeros(ctx, {4, 5}, M);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      tci::set_elem(ctx, M, ElemCoors{static_cast<ElemCoor>(i), static_cast<ElemCoor>(j)},
                    expected_M(i, j));
    }
  }
  tci::transpose(ctx, M, {1, 0});

  // After transpose, logical M_t[a, b] == M_original[b, a]. Slice the
  // [1, 4) x [0, 3) sub-block of the transposed tensor.
  tci::extract_sub(ctx, M, tci::List<tci::Pair<ElemCoor, ElemCoor>>{{1, 4}, {0, 3}});

  REQUIRE(tci::shape(ctx, M).size() == 2);
  CHECK(tci::shape(ctx, M)[0] == 3);
  CHECK(tci::shape(ctx, M)[1] == 3);

  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) {
      // Sliced position (r, c) corresponds to M_t[1+r, 0+c] == M_original[c, 1+r].
      auto val
          = tci::get_elem(ctx, M, ElemCoors{static_cast<ElemCoor>(r), static_cast<ElemCoor>(c)});
      CHECK(val == doctest::Approx(expected_M(c, 1 + r)));
    }
  }

  tci::destroy_context(ctx);
}

TEST_CASE("replace_sub on transposed (non-contiguous) tensor") {
  tci::context_handle_t<Tensor> ctx;
  tci::create_context(ctx);

  // Same construction; this time replace a 2x2 block of the transposed
  // tensor and verify both replaced and untouched positions.
  Tensor M;
  tci::zeros(ctx, {4, 5}, M);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      tci::set_elem(ctx, M, ElemCoors{static_cast<ElemCoor>(i), static_cast<ElemCoor>(j)},
                    expected_M(i, j));
    }
  }
  tci::transpose(ctx, M, {1, 0});

  Tensor sub;
  tci::zeros(ctx, {2, 2}, sub);
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t c = 0; c < 2; ++c) {
      tci::set_elem(ctx, sub, ElemCoors{static_cast<ElemCoor>(r), static_cast<ElemCoor>(c)},
                    static_cast<Elem>(1000 + r * 10 + c));
    }
  }

  ElemCoors begin_pt = {1, 1};
  tci::replace_sub(ctx, M, sub, begin_pt);

  REQUIRE(tci::shape(ctx, M)[0] == 5);
  REQUIRE(tci::shape(ctx, M)[1] == 4);

  // Replaced positions: M_t[1..3, 1..3) == sub
  for (std::size_t a = 0; a < 5; ++a) {
    for (std::size_t b = 0; b < 4; ++b) {
      auto val
          = tci::get_elem(ctx, M, ElemCoors{static_cast<ElemCoor>(a), static_cast<ElemCoor>(b)});
      bool inside = (a >= 1 && a < 3 && b >= 1 && b < 3);
      if (inside) {
        Elem expected = static_cast<Elem>(1000 + (a - 1) * 10 + (b - 1));
        CHECK(val == doctest::Approx(expected));
      } else {
        // Untouched: M_t[a, b] == M_original[b, a]
        CHECK(val == doctest::Approx(expected_M(b, a)));
      }
    }
  }

  tci::destroy_context(ctx);
}
