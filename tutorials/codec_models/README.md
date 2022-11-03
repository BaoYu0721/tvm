# TVM 的一些使用

## 关于 TVM TensorRT Integration
可参考：https://tvm.apache.org/docs/how_to/deploy/tensorrt.html
1. 可通过设置环境变量 "TVM_TENSORRT_CACHE_DIR" 来设置 tensorrt engine 文件的存储位置。
2. 在使用 TVM TensorRT Integration 时，如果想将有的算子设为使用 tensorrt 的实现，有的则想使用 tvm 自己生成的 cuda 算子，这时可以打开 python/tvm/relay/op/contrib/tensorrt.py，里面的 "partition_for_tensorrt" 函数是用户端必须要调用的一个函数，在这个函数中，可以看到有一个 Sequential Pass，其中有一个 MergeComposite Pass，MergeComposite Pass 会传入一个 "pattern_table"；这时我们跳转到 "pattern_table" 函数定义，可以看到这里定义了很多 tensorrt 算子和 relay 算子的对应关系，Tuple 中的 "make_predicate(xxx_checker)" 就是用来检测当前 relay 算子是否符合 tensorrt 转换标准的，当我们想 disable 此算子不被 tensorrt 所转换时，我们只需修改这里的 checker，使得该 checker 永远返回 False 即可。