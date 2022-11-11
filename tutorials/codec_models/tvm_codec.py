import onnx
from onnx import numpy_helper
import numpy as np
import os
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt

import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm

from common import parseCfgYaml, prepareData

def runRelayFrontEnd(onnx_model, in_shape_dict):
    mod, params = relay.frontend.from_onnx(onnx_model, in_shape_dict)
    return mod, params

def createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path, pass_ctx_config=None):
    # with tvm.transform.PassContext(opt_level=3, config={"relay.FuseOps.max_depth": 1}):
    with tvm.transform.PassContext(opt_level=3, config=pass_ctx_config):
        lib = relay.build(mod, target=target, params=params)
    if save_lib_path is not None:
        lib.export_library(save_lib_path)
    dev = tvm.device(str(target), 0)
    if debug_flag:
        module = GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.graph_json, dump_root=debug_dir)
    else:
        module = graph_executor.GraphModule(lib['default'](dev))
    return module

def runGraph(module, in_data_dict : dict, out_num : int):
    for in_name, data in in_data_dict.items():
        module.set_input(in_name, data)
    module.run()
    tvm_out_list = []
    for i in range(out_num):
        tvm_out_list.append(module.get_output(i).numpy())
    return tvm_out_list

def getAutoTVMTuningOption(model_name, target):
    autotvm_logname = './tune_logs/autotvm_{}_{}.tnlg'.format(model_name, target)
    graph_opt_filename = './tune_logs/autotvm_{}_{}_graph_opt.tnlg'.format(model_name, target)
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

def autoSchedulerTune(log_file, target, mod, params, debug_flag):
    tasks, task_weights = auto_scheduler.extract_tasks(mod['main'], params, target)
    if debug_flag:
        for idx, task in enumerate(tasks):
            print ('========== Task %d  (workload key: %s) ==========' % (idx, task.workload_key))
            print (task.compute_dag)

    if target == 'cuda':
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=2, min_repeat_ms=300, timeout=10)
        as_runner = measure_ctx.runner
    elif target == 'llvm':
        as_runner = auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True)
    else:
        as_runner = None
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=900*len(tasks),
        # num_measure_trials=4,
        runner=as_runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)

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
    cfg = parseCfgYaml('./config.yaml')
    tune_flag, use_tensorrt = cfg['use_tvm_tune'], cfg['use_tvm_trt_integration']
    debug_flag, save_lib_flag, tune_method = False, True, cfg['tvm_tune_method']

    model_name, target, data_idx = cfg['model_name'], cfg['target'], cfg['data_idx']
    use_tensorrt = use_tensorrt and (target == 'cuda')  # 只有在 cuda 下才可以考虑用 tensorrt
    debug_dir = './debug_{}'.format(model_name)
    onnx_model = onnx.load(cfg[model_name]['model_path'])
    if save_lib_flag:
        save_lib_path = './libs/{}.so'.format(model_name)
    else:
        save_lib_path = None
    checkOnnxModel(onnx_model)

    in_shape_dict, in_data_dict, out_shape_dict, out_data_dict = prepareData(cfg)

    mod, params = runRelayFrontEnd(onnx_model, in_shape_dict)
    if use_tensorrt:
        mod = partition_for_tensorrt(mod, params)
        print(mod['main'])
    tvm_out_list, pass_ctx_config = None, None

    if tune_flag:
        if tune_method == 'autotvm':
            tuning_option = getAutoTVMTuningOption(model_name, target)
            if tuning_option is None:
                raise ValueError('target not supported!')

            autoTVMTuneKernel(target, mod, params, tuning_option)
            with autotvm.apply_history_best(tuning_option['log_filename']):
                module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path)
                tvm_out_list = runGraph(module, in_data_dict, len(out_shape_dict))

            if tuning_option['use_graph_tuning']:
                autoTVMTuneGraph(mod, target, in_shape_dict, tuning_option)
                with autotvm.apply_graph_best(tuning_option['graph_opt_filename']):
                    module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path)
                    tvm_out_list = runGraph(module, in_data_dict, len(out_shape_dict))
        else:
            # use auto_scheduler
            log_file = './tune_logs/autoscheduler_{}_{}.tnlg'.format(model_name, target)
            autoSchedulerTune(log_file, target, mod, params, debug_flag)

            with auto_scheduler.ApplyHistoryBest(log_file):
                pass_ctx_config = {'relay.backend.use_auto_scheduler': True}
                module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path, pass_ctx_config)
                tvm_out_list = runGraph(module, in_data_dict, len(out_shape_dict))
    else:
        # 不 tune performance， 直接运行
        module = createGraph(target, mod, params, debug_flag, debug_dir, save_lib_path)
        tvm_out_list = runGraph(module, in_data_dict, len(out_shape_dict))

    if tvm_out_list is not None:
        i = 0
        for out_name, out_data in out_data_dict.items():
            err = np.abs(tvm_out_list[i].astype(np.float32) - out_data.astype(np.float32))
            print ('the {} th output:'.format(i), err.max(), err.min(), err.mean())
            i += 1
