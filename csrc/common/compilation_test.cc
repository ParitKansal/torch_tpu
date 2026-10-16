/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/common/compilation.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "csrc/common/compilation_spec.h"
#include "csrc/common/compilation_test_helper.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/excess_precision.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/pjrt/pjrt_state.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {
namespace {

using testing::ElementsAre;

class MakeCompilerOptionsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // This must be done before MakeCompilerOptions() is called, as the latter
    // depends on the PjRt client.
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }
};

UniqueCompileOptions GetCompileOptions(const CompilationMode mode) {
  return GetCompilationSpec(mode).xla_compile_options;
}

TEST_F(MakeCompilerOptionsTest, DefaultToO1ForEagerMode) {
  UniqueCompileOptions options =
      GetCompileOptions(CompilationMode::kFastCompile);
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_O1);
}

TEST_F(MakeCompilerOptionsTest, DefaultToUnsetForTorchCompileMode) {
  UniqueCompileOptions options =
      GetCompileOptions(CompilationMode::kFastRuntime);
  EXPECT_EQ(options->executable_build_options.optimization_level(),
            xla::ExecutionOptions::EFFORT_UNKNOWN);
}

TEST_F(MakeCompilerOptionsTest, CompilerOptionOverrides) {
  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O1"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_EQ(options->executable_build_options.optimization_level(),
              xla::ExecutionOptions::EFFORT_O1);
  }

  {
    ScopedCompilerOptionOverrides overrides({{"xla_optimization_level", "O2"}});
    UniqueCompileOptions options =
        GetCompileOptions(CompilationMode::kFastCompile);
    EXPECT_EQ(options->executable_build_options.optimization_level(),
              xla::ExecutionOptions::EFFORT_O2);
  }
}

TEST_F(MakeCompilerOptionsTest, LocksExcessPrecision) {
  // Ensure the default compilation context is initialized hermetically.
  GetCompileOptions(CompilationMode::kFastCompile);
  EXPECT_TRUE(IsAllowExcessPrecisionLocked());
  const bool current = GetAllowExcessPrecision();
  SetAllowExcessPrecision(current);
  EXPECT_THROW(SetAllowExcessPrecision(!current), torch_tpu::TtError);
}

class TupleArgumentsTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    PjrtBackend::GetInstance().SetPjRtInitializationOptions(
        {.device_type = "tpu"});
    ABSL_CHECK_OK(PjrtBackend::GetInstance().EnsureInitialized());
  }

  void SetUp() override {
    TT_ASSERT_OK_AND_ASSIGN(client_, PjrtBackend::GetInstance().GetClient());
  }

  xla::PjRtClient* client_ = nullptr;
};

// Returns the parameter shapes of the entry computation of `executable`.
std::vector<xla::Shape> GetEntryParameterShapes(
    const xla::PjRtLoadedExecutable& executable) {
  auto modules = executable.GetHloModules();
  ABSL_CHECK_OK(modules);             // CRASH_OK
  ABSL_CHECK_EQ(modules->size(), 1);  // CRASH_OK
  const xla::ComputationLayout& layout =
      (*modules)[0]->entry_computation_layout();
  std::vector<xla::Shape> shapes;
  for (int i = 0; i < layout.parameter_count(); ++i) {
    shapes.push_back(layout.parameter_shape(i));
  }
  return shapes;
}

TEST_F(TupleArgumentsTest, DefaultThresholdMatchesJax) {
  EXPECT_EQ(GetTupleArgumentsThreshold(), 2000);
}

TEST_F(TupleArgumentsTest, TuplesArgumentsAboveThreshold) {
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_, MakeSumOfArgumentsModule(5),
                               xla::CompileOptions(),
                               /*tuple_arguments_threshold=*/4));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), 1);
  ASSERT_TRUE(parameter_shapes[0].IsTuple());
  EXPECT_EQ(parameter_shapes[0].tuple_shapes().size(), 5);
}

TEST_F(TupleArgumentsTest, KeepsArgumentsFlatAtThreshold) {
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_, MakeSumOfArgumentsModule(4),
                               xla::CompileOptions(),
                               /*tuple_arguments_threshold=*/4));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), 4);
  for (const xla::Shape& shape : parameter_shapes) {
    EXPECT_TRUE(shape.IsArray());
  }
}

TEST_F(TupleArgumentsTest, TuplesExplicitArgumentLayouts) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {128, 128});
  *shape.mutable_layout() = xla::LayoutUtil::MakeLayout({1, 0});
  xla::CompileOptions options;
  options.argument_layouts = std::vector<xla::Shape>(5, shape);

  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      CompileAndLoadMlirModule(*client_, MakeSumOfArgumentsModule(5),
                               std::move(options),
                               /*tuple_arguments_threshold=*/4));

  const std::vector<xla::Shape> parameter_shapes =
      GetEntryParameterShapes(*executable);
  ASSERT_EQ(parameter_shapes.size(), 1);
  ASSERT_TRUE(parameter_shapes[0].IsTuple());
  EXPECT_EQ(parameter_shapes[0].tuple_shapes().size(), 5);
}

TEST_F(TupleArgumentsTest, ReportsParameterDeviceShapesOfTupledArguments) {
  xla::CompileOptions options;
  options.parameter_is_tupled_arguments = true;
  TT_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable,
      client_->CompileAndLoad(MakeSumOfArgumentsModule(3), std::move(options)));
  TT_ASSERT_OK_AND_ASSIGN(
      SharedLoadedExecutableWithMetadata executable,
      LoadedExecutableWithMetadata::MakeShared(std::move(loaded_executable)));

  // Argument layouts can only be validated if every argument has a shape.
  const std::vector<std::optional<xla::Shape>>& shapes =
      executable->parameter_device_shapes();
  ASSERT_EQ(shapes.size(), 3);
  for (const std::optional<xla::Shape>& shape : shapes) {
    ASSERT_TRUE(shape.has_value());
    EXPECT_THAT(shape->dimensions(), ElementsAre(128, 128));
    EXPECT_TRUE(shape->has_layout());
  }
}

}  // namespace
}  // namespace torch_tpu
