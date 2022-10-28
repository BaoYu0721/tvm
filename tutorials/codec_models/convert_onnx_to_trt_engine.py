import onnx
from typing import Any, List, Dict, Set
from onnx import ModelProto, ValueInfoProto
import onnx.checker
import os
import subprocess

from tvm_codec import parseSimpleCfg, model_data_map, parseDataShapeFromTxt, model_inname_map

def update_inputs_outputs_dims(model: ModelProto, input_dims: Dict[str, List[Any]], output_dims: Dict[str, List[Any]]) -> ModelProto:
    """
        This function updates the dimension sizes of the model's inputs and outputs to the values
        provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
        will be set for that dimension.

        Example. if we have the following shape for inputs and outputs:
                shape(input_1) = ('b', 3, 'w', 'h')
                shape(input_2) = ('b', 4)
                and shape(output)  = ('b', 'd', 5)

            The parameters can be provided as:
                input_dims = {
                    "input_1": ['b', 3, 'w', 'h'],
                    "input_2": ['b', 4],
                }
                output_dims = {
                    "output": ['b', -1, 5]
                }

            Putting it together:
                model = onnx.load('model.onnx')
                updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
                onnx.save(updated_model, 'model.onnx')
    """
    dim_param_set: Set[str] = set()

    def init_dim_param_set(dim_param_set: Set[str], value_infos: List[ValueInfoProto]) -> None:
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField('dim_param'):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    def update_dim(tensor: ValueInfoProto, dim: Any, j: int, name: str) -> None:
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + '_' + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError('Unable to generate unique dim_param for axis {} of {}. Please manually provide a dim_param value.'
                        .format(j, name))
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError(f'Only int or str is accepted as dimension value, incorrect type: {type(dim)}')

    for input in model.graph.input:
        input_name = input.name
        input_dim_arr = input_dims[input_name]
        for j, dim in enumerate(input_dim_arr):
            update_dim(input, dim, j, input_name)

    for output in model.graph.output:
        output_name = output.name
        output_dim_arr = output_dims[output_name]
        for j, dim in enumerate(output_dim_arr):
            update_dim(output, dim, j, output_name)

    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    model_name, target, model_version, data_idx = parseSimpleCfg('./simple_cfg.txt')
    dir_name = model_data_map[model_name]
    input_shape = parseDataShapeFromTxt('{}/{}/input_shape_{}.txt'.format(dir_name, model_version, data_idx))
    output_shape = parseDataShapeFromTxt('{}/{}/output_shape_{}.txt'.format(dir_name, model_version, data_idx))
    out_name = '{}_out'.format(model_name) if (model_name != 'z_decoder_int') else 'z_decoder_out'

    model = onnx.load('./models/{}_onnx/{}.onnx'.format(model_version, model_name))
    changed_shape_mod = update_inputs_outputs_dims(model, {model_inname_map[model_name]: [*input_shape]}, {out_name: [*output_shape]})
    tmp_trt_dir = './tmp_trt_models'
    os.makedirs(tmp_trt_dir, exist_ok=True)
    tmp_onnx_path = '{}/{}.onnx'.format(tmp_trt_dir, model_name)
    onnx.save_model(changed_shape_mod, tmp_onnx_path)
    tmp_engine_path = '{}/{}.engine'.format(tmp_trt_dir, model_name)

    # trt_cmd = ['trtexec', '--onnx='+tmp_onnx_path, '--workspace=64', '--buildOnly', '--saveEngine='+tmp_engine_path]
    # with subprocess.Popen(trt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
    #     print(proc.stdout.readlines())
    #     print(proc.stderr.readlines())
