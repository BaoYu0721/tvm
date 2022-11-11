import numpy as np
import yaml

def parseCfgYaml(yaml_path):
    with open(yaml_path, 'r') as fp:
        cfg = yaml.load(fp.read())
    return cfg

def parseDataShapeFromTxt(file_name):
    with open(file_name, 'r') as fp:
        content = fp.read().strip()
        shape = eval(content)
        return shape

def wildcardToSpecIdx(fn_with_wc : str, idx) -> str:
    res = fn_with_wc.split('*')
    assert(len(res) == 2)
    return res[0] + str(idx) + res[1]

def prepareData(cfg):
    model_name = cfg['model_name']
    dir_name = cfg[model_name]['data_dir']
    data_idx = cfg['data_idx']
    in_shape_dict, in_data_dict = {}, {}
    for i in range(len(cfg[model_name]['in_tensor_names'])):
        in_type = cfg[model_name]['in_types'][i]
        in_tensor_name = cfg[model_name]['in_tensor_names'][i]
        in_bin_file = wildcardToSpecIdx(cfg[model_name]['in_bin_names'][i], data_idx)
        in_shape_file = wildcardToSpecIdx(cfg[model_name]['in_shape_names'][i], data_idx)
        input_np = np.fromfile('{}/{}'.format(dir_name, in_bin_file), dtype=in_type)
        input_shape = parseDataShapeFromTxt('{}/{}'.format(dir_name, in_shape_file))
        input_np = input_np.reshape(input_shape)
        in_shape_dict[in_tensor_name] = input_shape
        in_data_dict[in_tensor_name] = input_np
    out_shape_dict, out_data_dict = {}, {}
    for i in range(len(cfg[model_name]['in_tensor_names'])):
        out_type = cfg[model_name]['out_types'][i]
        out_tensor_name = cfg[model_name]['out_tensor_names'][i]
        out_bin_file = wildcardToSpecIdx(cfg[model_name]['out_bin_names'][i], data_idx)
        out_shape_file = wildcardToSpecIdx(cfg[model_name]['out_shape_names'][i], data_idx)
        output_np = np.fromfile('{}/{}'.format(dir_name, out_bin_file), dtype=out_type)
        output_shape = parseDataShapeFromTxt('{}/{}'.format(dir_name, out_shape_file))
        output_np = output_np.reshape(output_shape)
        out_shape_dict[out_tensor_name] = output_shape
        out_data_dict[out_tensor_name] = output_np
    return in_shape_dict, in_data_dict, out_shape_dict, out_data_dict
