import onnx
from onnx import numpy_helper
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.debugger.debug_executor import GraphModuleDebug

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

model_data_map = {
    'y_decoder': 'ydecoder_inout',
    'y_encoder': 'yencoder_inout',
    'z_decoder_int': 'zdecoder_inout',
    'z_decoder': 'zdecoder_inout',
    'z_encoder': 'zencoder_inout'
}
model_type_map = {
    'y_decoder': np.float32,
    'y_encoder': np.float32,
    'z_decoder_int': np.int32,
    'z_decoder': np.int32,
    'z_encoder': np.float32
}
model_inname_map = {
    'y_decoder': 'y_decoder_data',
    'y_encoder': 'y_encoder_data',
    'z_decoder_int': 'z_decoder_data',
    'z_decoder': 'z_decoder_data',
    'z_encoder': 'z_encoder_data'
}

def parseDataShapeFromTxt(file_name):
    with open(file_name, 'r') as fp:
        content = fp.read().strip()
        shape = eval(content)
        return shape

def prepareData(model_name):
    dir_name = model_data_map[model_name]
    dtype = model_type_map[model_name]
    input_np = np.fromfile('{}/input_0.bin'.format(dir_name), dtype=dtype)
    output_np = np.fromfile('{}/output_0.bin'.format(dir_name), dtype=dtype)
    input_shape = parseDataShapeFromTxt('{}/input_shape_0.txt'.format(dir_name))
    output_shape = parseDataShapeFromTxt('{}/output_shape_0.txt'.format(dir_name))
    input_np = input_np.reshape(input_shape) #.astype(np.uint8)
    output_np = output_np.reshape(output_shape) #.astype(np.uint8)
    return input_np, output_np, input_shape, output_shape

def runRelayFrontEnd(input_name, onnx_model, img_data):
    shape_dict = {input_name: img_data.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params

def createGraph(target, mod, params, debug_flag, debug_dir):
    # with tvm.transform.PassContext(opt_level=3, config={"relay.FuseOps.max_depth": 1}):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
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

def tuneModel(target, mod, params, tuning_record_name):
    number = 10
    repeat = 1
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
    timeout = 10  # in seconds
    runner = autotvm.LocalRunner(number=number, repeat=repeat, timeout=timeout, min_repeat_ms=min_repeat_ms, enable_cpu_cache_flush=True)
    tuning_option = {
        "tuner": "xgb",
        "trials": 10,
        "early_stopping": 10,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": tuning_record_name,
    }
    # begin by extracting the tasks from the onnx model
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type='rank')
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

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
    target, debug_flag, model_name = 'cuda', False, 'y_decoder'
    input_name = model_inname_map[model_name]
    debug_dir = './debug_{}'.format(model_name)
    onnx_model = onnx.load('./{}.onnx'.format(model_name))
    checkOnnxModel(onnx_model)

    input_np, output_np, input_shape, output_shape = prepareData(model_name)

    mod, params = runRelayFrontEnd(input_name, onnx_model, input_np)

    module = createGraph(target, mod, params, debug_flag, debug_dir)
    tvm_output = runGraph(input_name, module, input_np, output_shape)

    # tuning_record_name = "zdecoder_autotuning.json"
    # tuneModel(target, mod, params, tuning_record_name)
    # with autotvm.apply_history_best(tuning_record_name):
    #     module = createGraph(target, mod, params, debug_flag, debug_dir)
    # tvm_output = runGraph(input_name, module, input_np, output_shape)

    err = tvm_output.astype(np.float32) - output_np.astype(np.float32)
    print (err.max(), err.min(), err.mean())
    # print (tvm_output)
    # print ('--------------------------------------')
    # print (output_np)
