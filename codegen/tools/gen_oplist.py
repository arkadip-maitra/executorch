# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

try:
    from executorch.codegen.parse import strip_et_fields
except ImportError:
    # If we build from source, executorch.codegen is not available.
    # We can use relative import instead.
    from ..parse import strip_et_fields

from torchgen.gen import LineLoader, parse_native_yaml_struct
from torchgen.selective_build.operator import SelectiveBuildOperator
from torchgen.selective_build.selector import merge_et_kernel_metadata

# Output YAML file format:
# ------------------------
#
# <BEGIN FILE CONTENTS>
# include_all_non_op_selectives: False
# include_all_operators: False
# debug_info:
#   - model1@v100
#   - model2@v50
# operators:
#   aten::add:
#     is_root_operator: Yes
#     is_used_for_training: Yes
#     include_all_overloads: No
#     debug_info:
#       - model1@v100
#       - model2@v50
#   aten::add.int:
#     is_root_operator: No
#     is_used_for_training: No
#     include_all_overloads: Yes
# et_kernel_metadata:
#   aten::add.out:
#     # A list of different kernel keys (tensors with dtype-enum/dim-order) combinations used in model
#       - v1/6;0,1|6;0,1|6;0,1|6;0,1  # Float, 0, 1
#       - v1/3;0,1|3;0,1|3;0,1|3;0,1  # Int, 0, 1
#   aten::mul.out:
#       - v1/6;0,1|6;0,1|6;0,1|6;0,1  # Float, 0, 1
# <END FILE CONTENTS>


class ScalarType(IntEnum):
    Byte = 0
    Char = 1
    Short = 2
    Int = 3
    Long = 4
    Float = 6
    Double = 7
    Bool = 11
    # TODO(jakeszwe): Verify these are unused and then remove support
    QInt8 = 12
    QUInt8 = 13
    QInt32 = 14
    QUInt4X2 = 16
    QUInt2X4 = 17
    # Types currently not implemented.
    # Half = 5
    # ComplexHalf = 8
    # ComplexFloat = 9
    # ComplexDouble = 10
    # BFloat16 = 15


class KernelType(IntEnum):
    TENSOR = 5
    TENSOR_LIST = 10
    OPTIONAL_TENSOR_LIST = 11


@dataclass
class SelectSpecInputs:
    root_ops_values: List[str] = field(default_factory=list)
    model_file_paths: List[str] = field(default_factory=list)
    ops_schema_yaml_paths: List[str] = field(default_factory=list)
    ops_dict_payloads: List[Dict[str, List[str]]] = field(default_factory=list)
    include_all_operators: bool = False
    source_name: Optional[str] = None


def _get_operators(model_file: str) -> List[str]:
    from executorch.codegen.tools.selective_build import (  # type: ignore[import-not-found]
        _get_program_from_buffer,
        _get_program_operators,
    )

    print("Processing model file: ", model_file)
    with open(model_file, "rb") as f:
        buf = f.read()

    program = _get_program_from_buffer(buf)
    operators = _get_program_operators(program)
    print(f"Model file loaded, operators are: {operators}")
    return operators


def _get_kernel_metadata_for_model(model_file: str) -> Dict[str, List[str]]:
    from executorch.codegen.tools.selective_build import (  # type: ignore[import-not-found]
        _get_io_metadata_for_program_operators,
        _get_program_from_buffer,
        _IOMetaData,
    )

    with open(model_file, "rb") as f:
        buf = f.read()

    program = _get_program_from_buffer(buf)
    operators_with_io_metadata = _get_io_metadata_for_program_operators(program)

    op_kernel_key_list: Dict[str, List[str]] = {}

    specialized_kernels: Set[List[_IOMetaData]]
    for op_name, specialized_kernels in operators_with_io_metadata.items():
        print(op_name)
        if op_name not in op_kernel_key_list:
            op_kernel_key_list[op_name] = []

        for specialized_kernel in specialized_kernels:
            version = "v1"
            kernel_key = version + "/"
            for io_metadata in specialized_kernel:
                if io_metadata.kernel_type in [
                    KernelType.TENSOR,
                    KernelType.TENSOR_LIST,
                    KernelType.OPTIONAL_TENSOR_LIST,
                ]:
                    dim_order = ",".join(map(str, io_metadata.dim_order))
                    kernel_key += f"{io_metadata.dtype};{dim_order}|"
            op_kernel_key_list[op_name].append(kernel_key[:-1])

    return op_kernel_key_list


def _get_et_kernel_metadata_from_ops_yaml(ops_yaml_path: str) -> Dict[str, List[str]]:
    ops = []
    with open(ops_yaml_path, "r") as f:
        es = yaml.load(f, Loader=LineLoader)
        func_entries = []
        for e in es:
            if "op" in e:
                ops.append(("aten::" if "::" not in e.get("op") else "") + e.get("op"))
            else:
                func_entries.append(e)
        strip_et_fields(es)
        parsed_yaml = parse_native_yaml_struct(
            func_entries, set(), None, path=ops_yaml_path, skip_native_fns_gen=True
        )
    ops.extend([f"{f.namespace}::{f.func.name}" for f in parsed_yaml.native_functions])
    # TODO (larryliu): accept the new op yaml syntax
    return {op: ["default"] for op in ops}


def _dump_yaml(
    op_list: List[str],
    output_path: Path,
    model_name: Optional[str] = None,
    et_kernel_metadata: Optional[Dict[str, List[str]]] = None,
    include_all_operators: bool = False,
):
    # no debug info yet
    output: dict[str, Any] = {}
    operators: Dict[str, Dict[str, object]] = {}
    for op_name in op_list:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [model_name],
            },
        )
        operators[op_name] = op.to_dict()

    output["operators"] = operators
    output["custom_classes"] = []
    output["build_features"] = []
    output["include_all_non_op_selectives"] = False
    output["include_all_operators"] = include_all_operators
    output["kernel_metadata"] = {}
    output["et_kernel_metadata"] = et_kernel_metadata
    with open(output_path, "wb") as out_file:
        out_file.write(
            yaml.safe_dump(
                output,
                default_flow_style=False,
            ).encode("utf-8")
        )


def create_kernel_key(maybe_kernel_key: str) -> str:
    # It is a kernel key.
    if maybe_kernel_key.lstrip().startswith("v1"):
        return maybe_kernel_key
    # It is a dtype.
    else:
        # Generate a kernel key based on the dtype provided.
        # Note: no dim order is included in this kernel key.
        # For a description of the kernel key format, see
        # executorch/blob/main/runtime/kernel/operator_registry.h#L97-L123
        try:
            dtype = ScalarType[maybe_kernel_key]
            return "v1/" + str(dtype.value) + ";"
        except KeyError:
            raise Exception(f"Unknown dtype: {maybe_kernel_key}")


def _split_root_ops(root_ops: str) -> List[str]:
    if "," in root_ops:
        return list(filter(None, map(str.strip, root_ops.split(","))))
    return list(filter(None, map(str.strip, root_ops.split())))


def _normalize_ops_dict_payload(data: Any) -> Dict[str, List[str]]:
    if not isinstance(data, dict):
        raise ValueError("ops_dict must be a mapping of operators to metadata lists")

    normalized: Dict[str, List[str]] = {}
    for op_name, metadata in data.items():
        if not isinstance(op_name, str):
            raise ValueError("ops_dict keys must be strings")
        if not isinstance(metadata, list):
            raise ValueError(f"ops_dict entry for '{op_name}' must be a list")
        normalized[op_name] = [str(item) for item in metadata]
    return normalized


def _load_json_or_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def _require_existing_file(path: Path, context: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{context} must point to an existing file, got {path}")
    return path


def _looks_like_ops_dict(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and len(data) > 0
        and all(
            isinstance(op_name, str) and isinstance(metadata, list)
            for op_name, metadata in data.items()
        )
    )


def _is_structured_select_spec(data: Any) -> bool:
    return isinstance(data, dict) and any(
        key in data for key in ("selectors", "ops_dict", "include_all_operators")
    )


def _resolve_select_spec_path(value: str, base_dir: Optional[Path]) -> str:
    path = Path(value)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path)


def _parse_structured_select_spec(  # noqa: C901
    data: Dict[str, Any], select_spec_path: Optional[Path] = None
) -> SelectSpecInputs:
    allowed_keys = {"include_all_operators", "selectors", "ops_dict"}
    unknown_keys = set(data.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in select spec: {sorted(unknown_keys)}. "
            f"Supported keys are {sorted(allowed_keys)}"
        )

    include_all_operators = data.get("include_all_operators", False)
    if not isinstance(include_all_operators, bool):
        raise ValueError("include_all_operators must be a boolean")

    inputs = SelectSpecInputs(
        include_all_operators=include_all_operators,
        source_name=str(select_spec_path) if select_spec_path is not None else None,
    )
    base_dir = select_spec_path.parent if select_spec_path is not None else None

    selectors = data.get("selectors", [])
    if not isinstance(selectors, list):
        raise ValueError("selectors must be a list")
    for selector in selectors:
        if not isinstance(selector, dict):
            raise ValueError("each selector entry must be a mapping")

        selector_type = selector.get("type")
        selector_value = selector.get("value")
        if selector_type == "all":
            inputs.include_all_operators = True
            continue
        if selector_type == "list":
            if isinstance(selector_value, str):
                inputs.root_ops_values.append(selector_value)
            elif isinstance(selector_value, list):
                inputs.root_ops_values.append(
                    ",".join(str(item) for item in selector_value)
                )
            else:
                raise ValueError(
                    "list selector values must be a string or list of strings"
                )
            continue
        if selector_type == "model":
            if not isinstance(selector_value, str):
                raise ValueError("model selector value must be a string path")
            resolved_path = _require_existing_file(
                Path(_resolve_select_spec_path(selector_value, base_dir)),
                "model selector value",
            )
            inputs.model_file_paths.append(str(resolved_path))
            continue
        if selector_type == "yaml":
            if not isinstance(selector_value, str):
                raise ValueError("yaml selector value must be a string path")
            resolved_path = _require_existing_file(
                Path(_resolve_select_spec_path(selector_value, base_dir)),
                "yaml selector value",
            )
            inputs.ops_schema_yaml_paths.append(str(resolved_path))
            continue

        raise ValueError(
            f"Unsupported selector type '{selector_type}'. "
            "Supported types are: all, list, model, yaml"
        )

    if "ops_dict" in data:
        inputs.ops_dict_payloads.append(_normalize_ops_dict_payload(data["ops_dict"]))

    return inputs


def _load_ops_dict_path(ops_dict_path: str) -> Dict[str, List[str]]:
    path = Path(ops_dict_path)
    if not path.is_file():
        raise ValueError(
            f"The value for --ops_dict_path must be a valid file, got {path}"
        )

    loaded = _load_json_or_yaml(path)
    if _is_structured_select_spec(loaded):
        if "ops_dict" not in loaded:
            raise ValueError(
                f"The select spec at {path} does not define an ops_dict payload"
            )
        return _normalize_ops_dict_payload(loaded["ops_dict"])
    if _looks_like_ops_dict(loaded):
        return _normalize_ops_dict_payload(loaded)
    raise ValueError(
        f"The value for --ops_dict_path must point to an ops_dict mapping or select spec, got {path}"
    )


def parse_select_spec(select_spec: str) -> SelectSpecInputs:
    select_spec = select_spec.strip()
    if not select_spec:
        return SelectSpecInputs()

    if select_spec.lower() == "all":
        return SelectSpecInputs(include_all_operators=True)

    if select_spec.startswith("{"):
        loaded = json.loads(select_spec)
        if _is_structured_select_spec(loaded):
            return _parse_structured_select_spec(loaded)
        if _looks_like_ops_dict(loaded):
            return SelectSpecInputs(
                ops_dict_payloads=[_normalize_ops_dict_payload(loaded)]
            )
        raise ValueError(
            "Inline JSON select specs must be either a structured select spec "
            "or an ops_dict mapping"
        )

    candidate_path = Path(select_spec)
    suffix = candidate_path.suffix.lower()
    if suffix == ".pte":
        candidate_path = _require_existing_file(
            candidate_path, "The value for --select_spec"
        )
        return SelectSpecInputs(
            model_file_paths=[str(candidate_path)],
            source_name=str(candidate_path),
        )

    if suffix in {".json", ".yaml", ".yml"}:
        candidate_path = _require_existing_file(
            candidate_path, "The value for --select_spec"
        )
        loaded = _load_json_or_yaml(candidate_path)
        if _is_structured_select_spec(loaded):
            return _parse_structured_select_spec(loaded, candidate_path)
        if _looks_like_ops_dict(loaded):
            return SelectSpecInputs(
                ops_dict_payloads=[_normalize_ops_dict_payload(loaded)],
                source_name=str(candidate_path),
            )
        if suffix in {".yaml", ".yml"}:
            if isinstance(loaded, dict):
                raise ValueError(
                    "YAML select specs must be either a structured select spec, "
                    "a non-empty ops_dict mapping, or an operator schema YAML list"
                )
            return SelectSpecInputs(
                ops_schema_yaml_paths=[str(candidate_path)],
                source_name=str(candidate_path),
            )
        raise ValueError(
            f"JSON select specs at {candidate_path} must be either a structured select spec "
            "or an ops_dict mapping"
        )

    return SelectSpecInputs(root_ops_values=[select_spec])


def _has_dtype_selective_metadata(et_kernel_metadata: Dict[str, List[str]]) -> bool:
    for metadata_list in et_kernel_metadata.values():
        for metadata in metadata_list:
            if metadata != "default" and "/" in metadata:
                return True
    return False


def select_spec_supports_dtype_selective_build(select_spec: str) -> bool:
    inputs = parse_select_spec(select_spec)
    if inputs.include_all_operators:
        return False
    if inputs.model_file_paths or inputs.ops_dict_payloads:
        return True
    return False


def gen_oplist(  # noqa: C901
    output_path: Path,
    model_file_path: Optional[str] = None,
    ops_schema_yaml_path: Optional[str] = None,
    root_ops: Optional[str] = None,
    ops_dict: Optional[str] = None,
    ops_dict_path: Optional[str] = None,
    select_spec: Optional[str] = None,
    include_all_operators: bool = False,
    require_dtype_selective_metadata: bool = False,
):
    select_spec_inputs = (
        parse_select_spec(select_spec) if select_spec else SelectSpecInputs()
    )
    root_ops_values: List[str] = []
    if root_ops:
        root_ops_values.append(root_ops)
    root_ops_values.extend(select_spec_inputs.root_ops_values)

    ops_dict_payloads: List[Dict[str, List[str]]] = []
    if ops_dict:
        ops_dict_payloads.append(_normalize_ops_dict_payload(json.loads(ops_dict)))
    if ops_dict_path:
        ops_dict_payloads.append(_load_ops_dict_path(ops_dict_path))
    ops_dict_payloads.extend(select_spec_inputs.ops_dict_payloads)

    model_file_paths: List[str] = []
    if model_file_path:
        model_file_paths.append(model_file_path)
    model_file_paths.extend(select_spec_inputs.model_file_paths)

    ops_schema_yaml_paths: List[str] = []
    if ops_schema_yaml_path:
        ops_schema_yaml_paths.append(ops_schema_yaml_path)
    ops_schema_yaml_paths.extend(select_spec_inputs.ops_schema_yaml_paths)

    include_all_operators = (
        include_all_operators or select_spec_inputs.include_all_operators
    )

    if not (
        model_file_paths
        or ops_schema_yaml_paths
        or root_ops_values
        or ops_dict_payloads
        or include_all_operators
    ):
        # dump empty yaml file
        _dump_yaml([], output_path)
        return

    assert output_path, "Need to provide output_path for dumped yaml file."
    op_set = set()
    source_name = select_spec_inputs.source_name
    et_kernel_metadata = {}  # type: ignore[var-annotated]
    for root_ops_value in root_ops_values:
        selected_root_ops = set(_split_root_ops(root_ops_value))
        op_set.update(selected_root_ops)
        et_kernel_metadata = merge_et_kernel_metadata(
            et_kernel_metadata, {op: ["default"] for op in selected_root_ops}
        )
    for ops_and_metadata in ops_dict_payloads:
        for op, metadata in ops_and_metadata.items():
            op_set.update({op})
            op_metadata = (
                [create_kernel_key(x) for x in metadata]
                if len(metadata) > 0
                else ["default"]
            )
            et_kernel_metadata = merge_et_kernel_metadata(
                et_kernel_metadata, {op: op_metadata}
            )
    for model_file in model_file_paths:
        assert os.path.isfile(
            model_file
        ), f"The value for --model_file_path needs to be a valid file, got {model_file}"
        model_kernel_metadata = _get_kernel_metadata_for_model(model_file)
        op_set.update(_get_operators(model_file))
        if source_name is None:
            source_name = model_file
        et_kernel_metadata = merge_et_kernel_metadata(
            et_kernel_metadata, model_kernel_metadata
        )
    for ops_yaml_path in ops_schema_yaml_paths:
        assert os.path.isfile(
            ops_yaml_path
        ), f"The value for --ops_schema_yaml_path needs to be a valid file, got {ops_yaml_path}"
        yaml_kernel_metadata = _get_et_kernel_metadata_from_ops_yaml(ops_yaml_path)
        if source_name is None:
            source_name = ops_yaml_path
        et_kernel_metadata = merge_et_kernel_metadata(
            et_kernel_metadata,
            yaml_kernel_metadata,
        )
        op_set.update(yaml_kernel_metadata.keys())
    if require_dtype_selective_metadata and not _has_dtype_selective_metadata(
        et_kernel_metadata
    ):
        raise ValueError(
            "Dtype selective build requires model- or ops_dict-derived kernel metadata. "
            "LIST, YAML, and include-all selectors only generate default kernel metadata."
        )
    _dump_yaml(
        sorted(op_set),
        output_path,
        source_name,
        et_kernel_metadata,
        include_all_operators,
    )


def main(args: List[Any]) -> None:  # noqa: C901
    """This binary generates selected_operators.yaml which will be consumed by caffe2/torchgen/gen.py.
    It reads the model file, deserialize it and dumps all the operators into selected_operators.yaml so
    it can be used in gen.py.
    """
    parser = argparse.ArgumentParser(
        description="Generate operator list from a model file"
    )
    parser.add_argument(
        "--output_path",
        help=("The path to the output yaml file (selected_operators.yaml)"),
        required=False,
    )
    parser.add_argument(
        "--model_file_path",
        help=("Path to an executorch program"),
        required=False,
    )
    parser.add_argument(
        "--ops_schema_yaml_path",
        help=("Dump operator names from operator schema yaml path"),
        required=False,
    )
    parser.add_argument(
        "--root_ops",
        help=("A comma separated list of root operators used by the model"),
        required=False,
    )
    parser.add_argument(
        "--ops_dict",
        help=(
            "A json object containing operators and their associated dtype and dim order"
        ),
        required=False,
    )
    parser.add_argument(
        "--ops_dict_path",
        help=("Path to a JSON or YAML file containing an ops_dict mapping"),
        required=False,
    )
    parser.add_argument(
        "--select-spec",
        "--select_spec",
        help=(
            "A consolidated selector string or spec file path. Supports 'all', op lists, "
            ".pte paths, operator YAML paths, ops_dict JSON/YAML files, or structured "
            "select spec JSON/YAML files."
        ),
        required=False,
    )
    parser.add_argument(
        "--include-all-operators",
        "--include_all_operators",
        action="store_true",
        default=False,
        help="Set this flag to request inclusion of all operators (i.e. build is not selective).",
        required=False,
    )
    parser.add_argument(
        "--require-dtype-selective-metadata",
        "--require_dtype_selective_metadata",
        action="store_true",
        default=False,
        help=(
            "Fail if the resolved selection does not contain model- or ops_dict-derived "
            "kernel metadata suitable for dtype selective build."
        ),
        required=False,
    )
    parser.add_argument(
        "--check-select-spec-supports-dtype-selective-build",
        "--check_select_spec_supports_dtype_selective_build",
        action="store_true",
        default=False,
        help=(
            "Print TRUE or FALSE depending on whether --select_spec can supply "
            "model- or ops_dict-derived metadata for dtype selective build."
        ),
        required=False,
    )
    options = parser.parse_args(args)

    if options.check_select_spec_supports_dtype_selective_build:
        if not options.select_spec:
            parser.error("--select_spec is required when checking dtype support")
        print(
            "TRUE"
            if select_spec_supports_dtype_selective_build(options.select_spec)
            else "FALSE"
        )
        return

    if not options.output_path:
        parser.error("--output_path is required")

    # check if the output_path is a directory, then generate operators
    # under selected_operators.yaml
    if Path(options.output_path).is_dir():
        output_path = Path(options.output_path) / "selected_operators.yaml"
    else:
        output_path = Path(options.output_path)
    try:
        gen_oplist(
            output_path=output_path,
            model_file_path=options.model_file_path,
            ops_schema_yaml_path=options.ops_schema_yaml_path,
            root_ops=options.root_ops,
            ops_dict=options.ops_dict,
            ops_dict_path=options.ops_dict_path,
            select_spec=options.select_spec,
            include_all_operators=options.include_all_operators,
            require_dtype_selective_metadata=options.require_dtype_selective_metadata,
        )
    except Exception as e:
        command = ["python codegen/tools/gen_oplist.py"]
        if options.model_file_path:
            command.append(f"--model_file_path {options.model_file_path}")
        if options.ops_schema_yaml_path:
            command.append(f"--ops_schema_yaml_path {options.ops_schema_yaml_path}")
        if options.root_ops:
            command.append(f"--root_ops {options.root_ops}")
        if options.ops_dict:
            command.append(f"--ops_dict {options.ops_dict}")
        if options.ops_dict_path:
            command.append(f"--ops_dict_path {options.ops_dict_path}")
        if options.select_spec:
            command.append(f"--select_spec {options.select_spec}")
        if options.include_all_operators:
            command.append("--include-all-operators")
        if options.require_dtype_selective_metadata:
            command.append("--require-dtype-selective-metadata")
        repro_command = " ".join(command)
        raise RuntimeError(
            f"""Failed to generate selected_operators.yaml. Repro command:
            {repro_command}
            """
        ) from e


if __name__ == "__main__":
    main(sys.argv[1:])
