# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any, cast, Dict, List, TYPE_CHECKING
from unittest.mock import NonCallableMock, patch

import yaml

if TYPE_CHECKING:
    import codegen.tools.gen_oplist as gen_oplist
    from codegen.tools.gen_oplist import ScalarType
else:
    try:
        gen_oplist = cast(Any, import_module("codegen.tools.gen_oplist"))
    except ImportError:
        gen_oplist = cast(Any, import_module("executorch.codegen.tools.gen_oplist"))
    ScalarType = cast(Any, gen_oplist.ScalarType)


class TestGenOpList(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ops_schema_yaml = os.path.join(self.temp_dir.name, "test.yaml")
        with open(self.ops_schema_yaml, "w") as f:
            f.write(
                """
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
            """
            )

    @patch.object(gen_oplist, "_get_operators")
    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_wrong_path(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
    ) -> None:
        args = ["--output_path=wrong_path", "--model_file_path=path2"]
        with self.assertRaises(RuntimeError):
            gen_oplist.main(args)

    @patch.object(gen_oplist, "_get_kernel_metadata_for_model")
    @patch.object(gen_oplist, "_get_operators")
    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_valid_model_path(
        self,
        mock_get_kernel_metadata_for_model: NonCallableMock,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
    ) -> None:
        temp_file = tempfile.NamedTemporaryFile()
        args = [
            f"--output_path={os.path.join(self.temp_dir.name, 'output.yaml')}",
            f"--model_file_path={temp_file.name}",
        ]
        gen_oplist.main(args)
        mock_get_operators.assert_called_once_with(temp_file.name)
        temp_file.close()

    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_valid_root_ops(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::add,aten::mul",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add", "aten::mul"],
            Path(output_path),
            None,
            {"aten::add": ["default"], "aten::mul": ["default"]},
            False,
        )

    def test_parse_select_spec_detects_all(self) -> None:
        inputs = gen_oplist.parse_select_spec("all")
        self.assertTrue(inputs.include_all_operators)
        self.assertListEqual([], inputs.root_ops_values)
        self.assertListEqual([], inputs.model_file_paths)

    def test_parse_select_spec_detects_list_scalar(self) -> None:
        inputs = gen_oplist.parse_select_spec("aten::add.out,aten::mul.out")
        self.assertListEqual(["aten::add.out,aten::mul.out"], inputs.root_ops_values)
        self.assertFalse(inputs.include_all_operators)

    def test_parse_select_spec_detects_model_scalar(self) -> None:
        model_path = os.path.join(self.temp_dir.name, "model.pte")
        Path(model_path).touch()
        inputs = gen_oplist.parse_select_spec(model_path)
        self.assertListEqual([model_path], inputs.model_file_paths)
        self.assertEqual(model_path, inputs.source_name)

    def test_parse_select_spec_detects_ops_yaml_path(self) -> None:
        inputs = gen_oplist.parse_select_spec(self.ops_schema_yaml)
        self.assertListEqual([self.ops_schema_yaml], inputs.ops_schema_yaml_paths)
        self.assertEqual(self.ops_schema_yaml, inputs.source_name)

    def test_parse_select_spec_detects_structured_spec_file(self) -> None:
        spec_path = os.path.join(self.temp_dir.name, "select_spec.yaml")
        model_path = os.path.join(self.temp_dir.name, "model.pte")
        Path(model_path).touch()
        with open(spec_path, "w") as f:
            f.write(
                """
include_all_operators: false
selectors:
  - type: model
    value: model.pte
  - type: list
    value:
      - aten::relu.out
      - aten::sigmoid.out
  - type: yaml
    value: test.yaml
ops_dict:
  aten::add.out:
    - Float
"""
            )
        inputs = gen_oplist.parse_select_spec(spec_path)
        self.assertListEqual([model_path], inputs.model_file_paths)
        self.assertListEqual(
            ["aten::relu.out,aten::sigmoid.out"], inputs.root_ops_values
        )
        self.assertListEqual([self.ops_schema_yaml], inputs.ops_schema_yaml_paths)
        self.assertListEqual(
            [{"aten::add.out": ["Float"]}],
            inputs.ops_dict_payloads,
        )
        self.assertEqual(spec_path, inputs.source_name)

    def test_parse_select_spec_detects_ops_dict_json_file(self) -> None:
        ops_dict_path = os.path.join(self.temp_dir.name, "ops_dict.json")
        with open(ops_dict_path, "w") as f:
            json.dump({"aten::add.out": ["Float"], "aten::mm.out": []}, f)

        inputs = gen_oplist.parse_select_spec(ops_dict_path)
        self.assertListEqual(
            [{"aten::add.out": ["Float"], "aten::mm.out": []}],
            inputs.ops_dict_payloads,
        )
        self.assertEqual(ops_dict_path, inputs.source_name)

    def test_parse_select_spec_rejects_empty_inline_ops_dict(self) -> None:
        with self.assertRaises(ValueError):
            gen_oplist.parse_select_spec("{}")

    def test_parse_select_spec_rejects_missing_model_path(self) -> None:
        with self.assertRaises(ValueError):
            gen_oplist.parse_select_spec(
                os.path.join(self.temp_dir.name, "missing_model.pte")
            )

    def test_parse_select_spec_rejects_missing_yaml_path(self) -> None:
        with self.assertRaises(ValueError):
            gen_oplist.parse_select_spec(
                os.path.join(self.temp_dir.name, "missing_select_spec.yaml")
            )

    def test_parse_select_spec_rejects_yaml_mapping_without_selectors(self) -> None:
        yaml_path = os.path.join(self.temp_dir.name, "invalid_select_spec.yaml")
        with open(yaml_path, "w") as f:
            f.write("{}\n")

        with self.assertRaises(ValueError):
            gen_oplist.parse_select_spec(yaml_path)

    def test_select_spec_supports_dtype_selective_build_for_model(self) -> None:
        model_path = os.path.join(self.temp_dir.name, "model.pte")
        Path(model_path).touch()
        self.assertTrue(
            gen_oplist.select_spec_supports_dtype_selective_build(model_path)
        )

    def test_select_spec_supports_dtype_selective_build_for_yaml_is_false(self) -> None:
        self.assertFalse(
            gen_oplist.select_spec_supports_dtype_selective_build(self.ops_schema_yaml)
        )

    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_root_ops_and_dtypes(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        ops_dict = {
            "aten::add": ["v1/3;0,1|3;0,1|3;0,1|3;0,1", ScalarType.Float.name],
            "aten::mul": [],
        }
        args = [
            f"--output_path={output_path}",
            f"--ops_dict={json.dumps(ops_dict)}",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add", "aten::mul"],
            Path(output_path),
            None,
            {
                "aten::add": [
                    "v1/3;0,1|3;0,1|3;0,1|3;0,1",
                    "v1/6;",
                ],
                "aten::mul": ["default"],
            },
            False,
        )

    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_select_spec_all_scalar(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        gen_oplist.main(
            [
                f"--output_path={output_path}",
                "--select_spec=all",
            ]
        )
        mock_dump_yaml.assert_called_once_with(
            [],
            Path(output_path),
            None,
            {},
            True,
        )

    @patch.object(gen_oplist, "_get_operators")
    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_both_op_list_and_ops_schema_yaml_merges(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        test_path = os.path.join(self.temp_dir.name, "test.yaml")
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::relu.out",
            f"--ops_schema_yaml_path={self.ops_schema_yaml}",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add.out", "aten::mul.out", "aten::relu.out"],
            Path(output_path),
            test_path,
            {
                "aten::relu.out": ["default"],
                "aten::add.out": ["default"],
                "aten::mul.out": ["default"],
            },
            False,
        )

    @patch.object(gen_oplist, "_get_kernel_metadata_for_model")
    @patch.object(gen_oplist, "_get_operators")
    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_select_spec_structured_spec_merges_sources(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
        mock_get_kernel_metadata_for_model: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        spec_path = os.path.join(self.temp_dir.name, "select_spec.yaml")
        model_path = os.path.join(self.temp_dir.name, "model.pte")
        Path(model_path).touch()
        model_kernel_key = "v1/6;0,1|6;0,1|6;0,1|6;0,1"
        mock_get_operators.return_value = ["aten::mm.out", "aten::add.out"]
        mock_get_kernel_metadata_for_model.return_value = {
            "aten::add.out": [model_kernel_key],
            "aten::mm.out": [model_kernel_key],
        }
        with open(spec_path, "w") as f:
            f.write(
                """
include_all_operators: false
selectors:
  - type: model
    value: model.pte
  - type: list
    value: aten::relu.out
  - type: yaml
    value: test.yaml
ops_dict:
  aten::sigmoid.out:
    - Float
"""
            )

        gen_oplist.main(
            [
                f"--output_path={output_path}",
                f"--select_spec={spec_path}",
            ]
        )
        mock_dump_yaml.assert_called_once_with(
            [
                "aten::add.out",
                "aten::mm.out",
                "aten::mul.out",
                "aten::relu.out",
                "aten::sigmoid.out",
            ],
            Path(output_path),
            spec_path,
            {
                "aten::relu.out": ["default"],
                "aten::add.out": ["default", model_kernel_key],
                "aten::mul.out": ["default"],
                "aten::sigmoid.out": ["v1/6;"],
                "aten::mm.out": [model_kernel_key],
            },
            False,
        )

    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_ops_dict_path(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        ops_dict_path = os.path.join(self.temp_dir.name, "ops_dict.yaml")
        with open(ops_dict_path, "w") as f:
            f.write(
                """
aten::add.out:
  - Float
aten::mm.out: []
"""
            )

        gen_oplist.main(
            [
                f"--output_path={output_path}",
                f"--ops_dict_path={ops_dict_path}",
            ]
        )
        mock_dump_yaml.assert_called_once_with(
            ["aten::add.out", "aten::mm.out"],
            Path(output_path),
            None,
            {
                "aten::add.out": ["v1/6;"],
                "aten::mm.out": ["default"],
            },
            False,
        )

    def test_gen_op_list_with_select_spec_requires_dtype_metadata_fails_for_list(
        self,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        with self.assertRaises(RuntimeError):
            gen_oplist.main(
                [
                    f"--output_path={output_path}",
                    "--select_spec=aten::add.out,aten::mul.out",
                    "--require_dtype_selective_metadata",
                ]
            )

    def test_parse_select_spec_rejects_unknown_selector_type(self) -> None:
        spec_path = os.path.join(self.temp_dir.name, "bad_select_spec.yaml")
        with open(spec_path, "w") as f:
            f.write(
                """
selectors:
  - type: dict
    value: {}
"""
            )

        with self.assertRaises(ValueError):
            gen_oplist.parse_select_spec(spec_path)

    def test_parse_select_spec_rejects_missing_model_in_structured_spec(self) -> None:
        spec_path = os.path.join(self.temp_dir.name, "missing_model_spec.yaml")
        with open(spec_path, "w") as f:
            f.write(
                """
selectors:
  - type: model
    value: missing_model.pte
"""
            )

        with self.assertRaises(ValueError):
            gen_oplist.parse_select_spec(spec_path)

    @patch.object(gen_oplist, "_get_kernel_metadata_for_model")
    @patch.object(gen_oplist, "_get_operators")
    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_model_and_op_list_merges(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
        mock_get_kernel_metadata_for_model: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        temp_file = tempfile.NamedTemporaryFile()
        model_kernel_key = "v1/6;0,1|6;0,1|6;0,1|6;0,1"
        mock_get_operators.return_value = ["aten::mm.out", "aten::add.out"]
        mock_get_kernel_metadata_for_model.return_value = {
            "aten::add.out": [model_kernel_key],
            "aten::mm.out": [model_kernel_key],
        }
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::relu.out",
            f"--model_file_path={temp_file.name}",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add.out", "aten::mm.out", "aten::relu.out"],
            Path(output_path),
            temp_file.name,
            {
                "aten::relu.out": ["default"],
                "aten::add.out": [model_kernel_key],
                "aten::mm.out": [model_kernel_key],
            },
            False,
        )
        temp_file.close()

    @patch.object(gen_oplist, "_get_kernel_metadata_for_model")
    @patch.object(gen_oplist, "_get_operators")
    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_model_and_overlapping_op_list_widens_metadata(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
        mock_get_kernel_metadata_for_model: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        temp_file = tempfile.NamedTemporaryFile()
        model_kernel_key = "v1/6;0,1|6;0,1|6;0,1|6;0,1"
        mock_get_operators.return_value = ["aten::mm.out", "aten::add.out"]
        mock_get_kernel_metadata_for_model.return_value = {
            "aten::add.out": [model_kernel_key],
            "aten::mm.out": [model_kernel_key],
        }
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::add.out",
            f"--model_file_path={temp_file.name}",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add.out", "aten::mm.out"],
            Path(output_path),
            temp_file.name,
            {
                "aten::add.out": ["default", model_kernel_key],
                "aten::mm.out": [model_kernel_key],
            },
            False,
        )
        temp_file.close()

    @patch.object(gen_oplist, "_dump_yaml")
    def test_gen_op_list_with_include_all_operators(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::add,aten::mul",
            "--include_all_operators",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add", "aten::mul"],
            Path(output_path),
            None,
            {"aten::add": ["default"], "aten::mul": ["default"]},
            True,
        )

    def test_get_custom_build_selector_with_both_allowlist_and_yaml(
        self,
    ) -> None:
        op_list = ["aten::add", "aten::mul"]
        filename = os.path.join(self.temp_dir.name, "selected_operators.yaml")
        gen_oplist._dump_yaml(op_list, Path(filename), "model.pte")
        self.assertTrue(os.path.isfile(filename))
        with open(filename) as f:
            es = yaml.safe_load(f)
        ops = es["operators"]
        self.assertEqual(len(ops), 2)
        self.assertSetEqual(set(ops.keys()), set(op_list))

    def test_gen_oplist_generates_from_root_ops(
        self,
    ) -> None:
        filename = os.path.join(self.temp_dir.name, "selected_operators.yaml")
        op_list = ["aten::add.out", "aten::mul.out", "aten::relu.out"]
        comma = ","
        args = [
            f"--output_path={filename}",
            f"--root_ops={comma.join(op_list)}",
        ]
        gen_oplist.main(args)
        self.assertTrue(os.path.isfile(filename))
        with open(filename) as f:
            es = yaml.safe_load(f)
        ops = es["operators"]
        self.assertEqual(len(ops), 3)
        self.assertSetEqual(set(ops.keys()), set(op_list))

    def test_dump_operator_from_ops_schema_yaml(self) -> None:
        ops = gen_oplist._get_et_kernel_metadata_from_ops_yaml(self.ops_schema_yaml)
        self.assertListEqual(sorted(ops.keys()), ["aten::add.out", "aten::mul.out"])

    def test_dump_operator_from_ops_schema_yaml_with_op_syntax(self) -> None:
        ops_yaml = os.path.join(self.temp_dir.name, "ops.yaml")
        with open(ops_yaml, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- op: mul.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
            """
            )
        ops = gen_oplist._get_et_kernel_metadata_from_ops_yaml(ops_yaml)
        self.assertListEqual(sorted(ops.keys()), ["aten::add.out", "aten::mul.out"])

    def test_dump_operator_from_ops_schema_yaml_with_mix_syntax(self) -> None:
        mix_yaml = os.path.join(self.temp_dir.name, "mix.yaml")
        with open(mix_yaml, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
            """
            )
        ops = gen_oplist._get_et_kernel_metadata_from_ops_yaml(mix_yaml)
        self.assertListEqual(sorted(ops.keys()), ["aten::add.out", "aten::mul.out"])

    def test_get_kernel_metadata_from_ops_yaml(self) -> None:
        metadata: Dict[str, List[str]] = (
            gen_oplist._get_et_kernel_metadata_from_ops_yaml(self.ops_schema_yaml)
        )

        self.assertEqual(len(metadata), 2)

        self.assertIn("aten::add.out", metadata)
        # We only have one dtype/dim-order combo for add (float/0,1)
        self.assertEqual(len(metadata["aten::add.out"]), 1)
        self.assertEqual(
            metadata["aten::add.out"][0],
            "default",
        )

        self.assertIn("aten::mul.out", metadata)
        self.assertEqual(len(metadata["aten::mul.out"]), 1)
        self.assertEqual(
            metadata["aten::mul.out"][0],
            "default",
        )

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
