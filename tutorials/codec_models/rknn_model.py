from rknn.api import RKNN
import numpy as np
import os
import onnx
from onnx import version_converter

from common import parseCfgYaml, prepareData
from convert_onnx_to_trt_engine import update_inputs_outputs_dims

if __name__ == '__main__':
    cfg = parseCfgYaml('./config.yaml')
    model_name, target, data_idx = cfg['model_name'], cfg['target'], cfg['data_idx']
    in_shape_dict, in_data_dict, out_shape_dict, out_data_dict = prepareData(cfg)
    export_model_dir_name = './rknn_models'
    os.makedirs(export_model_dir_name, exist_ok=True)

    model = onnx.load(cfg[model_name]['model_path'])
    changed_shape_mod = update_inputs_outputs_dims(model, in_shape_dict, out_shape_dict)
    # rknn 需要 version 是 12 的 opset
    changed_shape_mod = version_converter.convert_version(changed_shape_mod, 12)
    tmp_onnx_path = '{}/{}.onnx'.format(export_model_dir_name, model_name)
    onnx.save_model(changed_shape_mod, tmp_onnx_path)

    rknn = RKNN(verbose=True)
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
    rknn.config(target_platform='rk3588')
    if (rknn.load_onnx(model=tmp_onnx_path) != 0):
        print('Load model failed!')
        exit(0)
    if (rknn.build(do_quantization=cfg['rknn_use_quant']) != 0):
        print('Build model failed!')
        exit(0)

    rknn_model_name = export_model_dir_name + '/' + model_name + '.rknn'
    if (rknn.export_rknn(rknn_model_name) != 0):
        print('Export rknn model failed!')
        exit(0)

    if (rknn.init_runtime() != 0):
    # if (rknn.init_runtime(target='rk3588', device_id='1b095ec0af5cfe69') != 0):
        print('Init runtime environment failed!')
        exit(0)

    input_list = []
    for in_name, in_data in in_data_dict.items():
        in_data = np.transpose(in_data, (0, 2, 3, 1))
        input_list.append(in_data)
    outputs = rknn.inference(inputs=input_list)
    i = 0
    for out_name, out_data in out_data_dict.items():
        # out_data = np.transpose(out_data, (0, 2, 3, 1))
        err = np.abs(outputs[i].astype(np.float32) - out_data.astype(np.float32))
        print ('the {} th output:'.format(i), err.max(), err.min(), err.mean())
        i += 1

    rknn.release()
