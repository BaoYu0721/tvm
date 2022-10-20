import onnx
from onnx import numpy_helper
import numpy as np
import os
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.debugger.debug_executor import GraphModuleDebug

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm

model_data_map = {
    'y_decoder': 'ydecoder_inout',
    'y_encoder': 'yencoder_inout',
    'z_decoder_int': 'zdecoder_inout',
    'z_decoder': 'zdecoder_inout',
    'z_encoder': 'zencoder_inout'
}
model_type_map = {
    'y_decoder': (np.float32, np.float32),
    'y_encoder': (np.float32, np.float32),
    'z_decoder_int': (np.int32, np.uint8),
    'z_decoder': (np.int32, np.uint8),
    'z_encoder': (np.float32, np.float32)
}
model_inname_map = {
    'y_decoder': 'y_decoder_data',
    'y_encoder': 'y_encoder_data',
    'z_decoder_int': 'z_decoder_data',
    'z_decoder': 'z_decoder_data',
    'z_encoder': 'z_encoder_data'
}

def parseSimpleCfg(cfg_fn):
    with open(cfg_fn, 'r') as fcfg:
        lines = fcfg.readlines()
        res = []
        for line in lines:
            res.append(line.strip())
        model_name, target, model_version = res[0], res[1], res[2]
    return model_name, target, model_version

def parseDataShapeFromTxt(file_name):
    with open(file_name, 'r') as fp:
        content = fp.read().strip()
        shape = eval(content)
        return shape

def prepareData(model_name, model_version, data_idx):
    dir_name = model_data_map[model_name]
    in_type, out_type = model_type_map[model_name]
    if model_version == 's0.4.1':
        in_type, out_type = np.float32, np.float32
    input_np = np.fromfile('{}/{}/input_{}.bin'.format(dir_name, model_version, data_idx), dtype=in_type)
    output_np = np.fromfile('{}/{}/output_{}.bin'.format(dir_name, model_version, data_idx), dtype=out_type)
    input_shape = parseDataShapeFromTxt('{}/{}/input_shape_{}.txt'.format(dir_name, model_version, data_idx))
    output_shape = parseDataShapeFromTxt('{}/{}/output_shape_{}.txt'.format(dir_name, model_version, data_idx))
    input_np = input_np.reshape(input_shape)
    output_np = output_np.reshape(output_shape)
    return input_np, output_np, input_shape, output_shape

def runRelayFrontEnd(input_name, onnx_model, img_data):
    shape_dict = {input_name: img_data.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params

def createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path):
    # with tvm.transform.PassContext(opt_level=3, config={"relay.FuseOps.max_depth": 1}):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    if save_lib_path is not None:
        lib.export_library(save_lib_path)
    dev = tvm.device(str(target), 0)
    if debug_flag:
        module = GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.graph_json, dump_root=debug_dir)
    else:
        module = graph_executor.GraphModule(lib['default'](dev))
    return module

def runGraph(input_name, module, img_data, output_shape):
    module.set_input(input_name, img_data)
    module.run()
    # tvm_output = module.get_output(0, tvm.nd.empty(output_shape, dtype='uint8')).numpy()
    tvm_output = module.get_output(0).numpy()
    return tvm_output

def getTuningOption(model_name, target):
    autotvm_logname = 'autotvm_{}_{}.log'.format(model_name, target)
    graph_opt_filename = 'autotvm_{}_{}_graph_opt.log'.format(model_name, target)
    tuning_option = None
    if target == 'cuda':
        tuning_option = {
            'log_filename': autotvm_logname,
            'tuner': 'xgb',
            'n_trial': 2000,
            'early_stopping': 600,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)
            ),
            'use_transfer_learning': True,
            'use_graph_tuning': False, # graph tuning is not implemented yet for cuda
            'use_DP': False,
            'graph_opt_filename': graph_opt_filename
        }
    elif target == 'llvm':
        tuning_option = {
            'log_filename': autotvm_logname,
            'tuner': 'random',
            'n_trial': 2000,
            'early_stopping': None,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True)
            ),
            'use_transfer_learning': False,
            'use_graph_tuning': True,
            'use_DP': False,
            'graph_opt_filename': graph_opt_filename
        }
    return tuning_option

def autoTVMTuneKernel(target, mod, params, tuning_option):
    tasks = autotvm.task.extract_from_program(mod['main'], params, target)
    tmp_log_file = tuning_option['log_filename'] + '.tmp'
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    for i, tsk in enumerate(reversed(tasks)):
        prefix = '[Task %2d/%2d] ' % (i + 1, len(tasks))
        if tuning_option['tuner'] == 'xgb' or tuning_option['tuner'] == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuning_option['tuner'] == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuning_option['tuner'] == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuning_option['tuner'] == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError('Invalid tuner: ' + tuning_option['tuner'])

        if tuning_option['use_transfer_learning']:
            if os.path.exists(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        tsk_trial = min(tuning_option['n_trial'], len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=tuning_option['early_stopping'],
            measure_option=tuning_option['measure_option'],
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file)
            ]
        )
    autotvm.record.pick_best(tmp_log_file, tuning_option['log_filename'])
    os.remove(tmp_log_file)

def autoTVMTuneGraph(mod, target, input_shapes, tuning_option):
    # 根据注释，TuneGraph 就是同时考虑 kernel 的运行时间和 layout 的转换时间
    target_ops = [
        relay.op.get('nn.conv2d'),
        relay.op.get('nn.conv2d_transpose'),
    ]
    GraphTuner = DPTuner if tuning_option['use_DP'] else PBQPTuner
    executor = GraphTuner(mod['main'], input_shapes, tuning_option['log_filename'], target_ops, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(tuning_option['graph_opt_filename'])

def checkOnnxModel(onnx_model):
    model_outname =[node.name for node in onnx_model.graph.output]
    input_name_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_name_all)  - set(input_initializer))
    print('Inputs: ', net_feed_input)
    print('Outputs: ', model_outname)
    initializers = onnx_model.graph.initializer
    weights = []
    for init in initializers:
        w = numpy_helper.to_array(init)
        weights.append(w)
        print (w.dtype)

if __name__ == '__main__':
    tune_flag = False
    # tune_method: 'autotvm' or 'autoscheduler'
    debug_flag, save_lib_flag, tune_method, data_idx = False, True, 'autotvm', 0

    model_name, target, model_version = parseSimpleCfg('./simple_cfg.txt')
    input_name = model_inname_map[model_name]
    debug_dir = './debug_{}'.format(model_name)
    onnx_model = onnx.load('./{}_onnx/{}.onnx'.format(model_version, model_name))
    if save_lib_flag:
        save_lib_path = './libs/{}.so'.format(model_name)
    else:
        save_lib_path = None
    checkOnnxModel(onnx_model)

    input_np, output_np, input_shape, output_shape = prepareData(model_name, model_version, data_idx)

    mod, params = runRelayFrontEnd(input_name, onnx_model, input_np)
    tvm_output = None

    if tune_flag:
        tuning_option = getTuningOption(model_name, target)
        if tuning_option is None:
            raise ValueError('target not supported!')

        autoTVMTuneKernel(target, mod, params, tuning_option)
        with autotvm.apply_history_best(tuning_option['log_filename']):
            module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path)
            tvm_output = runGraph(input_name, module, input_np, output_shape)

        if tuning_option['use_graph_tuning']:
            autoTVMTuneGraph(mod, target, {input_name: input_shape}, tuning_option)
            with autotvm.apply_graph_best(tuning_option['graph_opt_filename']):
                module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path)
                tvm_output = runGraph(input_name, module, input_np, output_shape)
    else:
        # 不 tune performance， 直接运行
        module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path)
        tvm_output = runGraph(input_name, module, input_np, output_shape)

    if tvm_output is not None:
        err = np.abs(tvm_output.astype(np.float32) - output_np.astype(np.float32))
        print (err.max(), err.min(), err.mean())
