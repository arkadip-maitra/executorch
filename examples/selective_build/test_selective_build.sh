#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of selective build for both the legacy selector
# knobs and the consolidated EXECUTORCH_SELECT_OPS API.
set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../.ci/scripts/utils.sh"


# BUCK2 examples; test internally in fbcode/xplat
# 1. `--config executorch.select_ops=all`: select all ops from the dependency
#       kernel libraries, register all of them into ExecuTorch runtime.
# 2. `--config executorch.select_ops=list`: Only select ops from `ops` kwarg
#       in `et_operator_library` macro.
# 3. `--config executorch.select_ops=yaml`: Only select from a yaml file from
#       `ops_schema_yaml_target` kwarg in `et_operator_library` macro
# 4. `--config executorch.select_ops=dict`: Only select ops from `ops_dict`
#       kwarg in `et_operator_library` macro. Add `dtype_selective_build = True`
#       to executorch_generated_lib to select dtypes specified in the dictionary.

# Other configs:
# - `--config executorch.max_kernel_num=N`: Only allocate memory for the
#       required number of operators. Users can retrieve N from `selected_operators.yaml`.

verify_selected_max_kernel_num_header() {
    local build_dir=$1
    echo "Verifying auto-right-sized MAX_KERNEL_NUM header was generated"
    local generated_header
    generated_header=$(find "${build_dir}" -name selected_max_kernel_num.h -print -quit)
    if [[ -z "${generated_header}" ]]; then
      echo "ERROR: selected_max_kernel_num.h not generated"
      exit 1
    fi
    grep -q "EXECUTORCH_SELECTED_MAX_KERNEL_NUM" "${generated_header}" \
      || { echo "ERROR: header missing expected define"; exit 1; }
    echo "Generated: $(sed -n '$p' "${generated_header}")"
}

verify_selected_op_variants_header() {
    local build_dir=$1
    echo "Verifying dtype-selective variant header was generated"
    local generated_variant_header
    generated_variant_header=$(find "${build_dir}" -name selected_op_variants.h -print -quit)
    if [[ -z "${generated_variant_header}" ]]; then
      echo "ERROR: selected_op_variants.h not generated"
      exit 1
    fi
}

test_buck2_select_all_ops() {
    echo "Exporting MobilenetV3"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="mv3"

    echo "Running selective build test"
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=all -- --model_path=./mv3.pte

    echo "Removing mv3.pte"
    rm "./mv3.pte"
}

test_buck2_select_ops_in_list() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="add_mul"

    echo "Running selective build test"
    # set max_kernel_num=22: 19 primops, add, mul
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.max_kernel_num=22 \
        --config=executorch.select_ops=list \
        -- --model_path=./add_mul.pte

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_buck2_select_ops_in_dict() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="add_mul"

    echo "Running selective build test"
    # select ops and their dtypes using the dictionary API.
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=dict \
        -- --model_path=./add_mul.pte

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_buck2_select_ops_from_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.portable.custom_ops.custom_ops_1

    echo "Running selective build test"
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=yaml -- --model_path=./custom_ops_1.pte

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

# CMake examples; test in OSS. Check the README for more information.
test_cmake_select_ops_in_list() {
    echo "Exporting MobilenetV2"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="mv2"

    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}
    # set MAX_KERNEL_NUM=22: 19 primops, add, mul
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE=Release \
            -DMAX_KERNEL_NUM=22 \
            -DEXECUTORCH_SELECT_OPS_LIST="aten::convolution.out,\
aten::_native_batch_norm_legit_no_training.out,aten::hardtanh.out,aten::add.out,\
aten::mean.out,aten::view_copy.out,aten::permute_copy.out,aten::addmm.out,\
aten,aten::clone.out" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./mv2.pte"

    echo "Removing mv2.pte"
    rm "./mv2.pte"
}

test_cmake_select_ops_in_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.portable.custom_ops.custom_ops_1
    local example_dir=examples/selective_build/advanced
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE=Release \
            -DEXECUTORCH_EXAMPLE_USE_CUSTOM_OPS=ON \
            -DEXECUTORCH_EXAMPLE_DEFINE_CUSTOM_TARGET=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./custom_ops_1.pte"

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

test_cmake_select_ops_in_model() {
    local model_name="add_mul"
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    # No -DMAX_KERNEL_NUM: selective build auto-right-sizes the registry from
    # the .pte via gen_selected_max_kernel_num().
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS_MODEL="./${model_export_name}" \
            -DEXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD=ON \
            -DEXECUTORCH_OPTIMIZE_SIZE=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    verify_selected_max_kernel_num_header "${build_dir}"
    verify_selected_op_variants_header "${build_dir}"

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./${model_export_name}"

    echo "Removing ${model_export_name}"
    rm "./${model_export_name}"
}

test_cmake_select_ops_consolidated_list() {
    local model_name="add_mul"
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_select_ops_list
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="aten::add.out,aten::mm.out" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir} with consolidated list selector"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    verify_selected_max_kernel_num_header "${build_dir}"

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./${model_export_name}"

    echo "Removing ${model_export_name}"
    rm "./${model_export_name}"
}

test_cmake_select_ops_consolidated_list_with_dtype_fails() {
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_select_ops_list_dtype_invalid
    rm -rf ${build_dir}

    echo "Verifying consolidated list selector rejects dtype selective build at configure time"
    if cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="aten::add.out,aten::mm.out" \
            -DEXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}; then
        echo "ERROR: consolidated list selector unexpectedly accepted dtype selective build"
        exit 1
    fi
}

test_cmake_select_ops_consolidated_model() {
    local model_name="add_mul"
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_select_ops_model
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="./${model_export_name}" \
            -DEXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD=ON \
            -DEXECUTORCH_OPTIMIZE_SIZE=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir} with consolidated model selector"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    verify_selected_max_kernel_num_header "${build_dir}"
    verify_selected_op_variants_header "${build_dir}"

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./${model_export_name}"

    echo "Removing ${model_export_name}"
    rm "./${model_export_name}"
}

test_cmake_select_ops_consolidated_model_from_subdir() {
    local model_name="add_mul"
    local model_export_name="${model_name}.pte"
    local repo_root
    repo_root=$(pwd)
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_select_ops_model_subdir
    rm -rf ${build_dir}

    echo "Configuring ${example_dir} from its subdirectory with a repo-root-relative consolidated model path"
    pushd "${example_dir}" >/dev/null
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="./${model_export_name}" \
            -DEXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD=ON \
            -DEXECUTORCH_OPTIMIZE_SIZE=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B"${repo_root}/${build_dir}" \
            .
    popd >/dev/null

    echo "Building ${example_dir} with repo-root-relative path normalization"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    verify_selected_max_kernel_num_header "${build_dir}"
    verify_selected_op_variants_header "${build_dir}"

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./${model_export_name}"

    echo "Removing ${model_export_name}"
    rm "./${model_export_name}"
}

test_cmake_select_ops_consolidated_structured_spec() {
    local model_name="add_mul"
    local spec_dir=cmake-out/selective_build_specs/structured
    local model_export_path="${spec_dir}/${model_name}.pte"
    local spec_path="${spec_dir}/select_spec.yaml"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_select_ops_structured
    echo "Exporting ${model_name}"
    rm -rf "${spec_dir}"
    mkdir -p "${spec_dir}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}" --output_dir="${spec_dir}"
    cat > "${spec_path}" <<EOF
include_all_operators: false
selectors:
  - type: model
    value: ./add_mul.pte
  - type: list
    value: aten::relu.out
EOF
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="${spec_path}" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir} with consolidated structured spec"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    echo "Verifying structured spec merged model and list selectors"
    local selected_operators_yaml=${build_dir}/executorch/executorch_selected_kernels/selected_operators.yaml
    if [[ ! -f "${selected_operators_yaml}" ]]; then
      echo "ERROR: selected_operators.yaml not generated"
      exit 1
    fi
    ${PYTHON_EXECUTABLE} - <<PY
import yaml

with open("${selected_operators_yaml}") as f:
    ops = set(yaml.safe_load(f)["operators"].keys())

expected = {"aten::add.out", "aten::mm.out", "aten::relu.out"}
missing = expected - ops
if missing:
    raise RuntimeError(f"Missing merged ops: {sorted(missing)}")
PY

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="${model_export_path}"

    echo "Removing structured spec artifacts"
    rm -rf "${spec_dir}"
}

test_cmake_select_ops_consolidated_dict_dtype() {
    local model_name="add_mul"
    local spec_dir=cmake-out/selective_build_specs/dict
    local model_export_path="${spec_dir}/${model_name}.pte"
    local spec_path="${spec_dir}/select_spec.json"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_select_ops_dict
    echo "Exporting ${model_name}"
    rm -rf "${spec_dir}"
    mkdir -p "${spec_dir}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}" --output_dir="${spec_dir}"
    cat > "${spec_path}" <<EOF
{
  "ops_dict": {
    "aten::add.out": ["Float"],
    "aten::mm.out": []
  }
}
EOF
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="${spec_path}" \
            -DEXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir} with consolidated dict+dtype spec"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    verify_selected_max_kernel_num_header "${build_dir}"
    verify_selected_op_variants_header "${build_dir}"

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="${model_export_path}"

    echo "Removing dict spec artifacts"
    rm -rf "${spec_dir}"
}

test_cmake_select_ops_in_model_and_list() {
    local model_name="add_mul"
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_combined
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS_MODEL="./${model_export_name}" \
            -DEXECUTORCH_SELECT_OPS_LIST="aten::relu.out" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir} with model and list selectors"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE

    echo "Verifying merged selected_operators.yaml contains both selector inputs"
    local selected_operators_yaml=${build_dir}/executorch/executorch_selected_kernels/selected_operators.yaml
    if [[ ! -f "${selected_operators_yaml}" ]]; then
      echo "ERROR: selected_operators.yaml not generated"
      exit 1
    fi
    ${PYTHON_EXECUTABLE} - <<PY
import yaml

with open("${selected_operators_yaml}") as f:
    ops = set(yaml.safe_load(f)["operators"].keys())

expected = {"aten::add.out", "aten::mm.out", "aten::relu.out"}
missing = expected - ops
if missing:
    raise RuntimeError(f"Missing merged ops: {sorted(missing)}")
PY

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./${model_export_name}"

    echo "Removing ${model_export_name}"
    rm "./${model_export_name}"
}

test_cmake_select_ops_mixed_legacy_and_consolidated_fails() {
    local model_name="add_mul"
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/selective_build/basic
    local build_dir=cmake-out/${example_dir}_mixed_api
    rm -rf ${build_dir}

    echo "Verifying consolidated API rejects legacy selector mixing"
    if cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS="./${model_export_name}" \
            -DEXECUTORCH_SELECT_OPS_MODEL="./${model_export_name}" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}; then
        echo "ERROR: mixed consolidated and legacy selectors unexpectedly succeeded"
        exit 1
    fi

    echo "Removing ${model_export_name}"
    rm "./${model_export_name}"
}

if [[ -z $BUCK ]];
then
  BUCK=buck2
fi

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ -z $CMAKE_BUILD_TYPE ]];
then
  CMAKE_BUILD_TYPE=Release
fi

if [[ $1 == "cmake" ]];
then
    cmake_install_executorch_lib $CMAKE_BUILD_TYPE
    test_cmake_select_ops_in_list
    test_cmake_select_ops_in_yaml
    test_cmake_select_ops_in_model
    test_cmake_select_ops_in_model_and_list
    test_cmake_select_ops_consolidated_list
    test_cmake_select_ops_consolidated_list_with_dtype_fails
    test_cmake_select_ops_consolidated_model
    test_cmake_select_ops_consolidated_model_from_subdir
    test_cmake_select_ops_consolidated_structured_spec
    test_cmake_select_ops_consolidated_dict_dtype
    test_cmake_select_ops_mixed_legacy_and_consolidated_fails
elif [[ $1 == "buck2" ]];
then
    test_buck2_select_all_ops
    test_buck2_select_ops_in_list
    test_buck2_select_ops_in_dict
    test_buck2_select_ops_from_yaml
fi
