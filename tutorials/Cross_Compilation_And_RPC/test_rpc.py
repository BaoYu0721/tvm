import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

def createExampleFuncAndData(mode):
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
    sch = te.create_schedule(B.op)

    if mode == 'remote_opencl':
        xo, xi = sch[B].split(B.op.axis[0], factor=32)
        sch[B].bind(xo, te.thread_axis("blockIdx.x"))
        sch[B].bind(xi, te.thread_axis("threadIdx.x"))

    return sch, [A, B]


if __name__ == '__main__':
    # mode: local_cpu, remote_cpu, remote_opencl
    mode = 'remote_opencl'

    if mode == 'local_cpu':
        target = "llvm"
    elif mode == 'remote_cpu':
        target = "llvm -mtriple=aarch64-linux-gnu"
    elif mode == 'remote_opencl':
        target = tvm.target.Target('opencl', host="llvm -mtriple=aarch64-linux-gnu")
    else:
        print ('unsupported mode!')
        exit(0)

    sch, var_list = createExampleFuncAndData(mode)
    func = tvm.build(sch, var_list, target=target, name="add_one")
    # save the lib at a local temp folder
    temp = utils.tempdir()
    local_path = temp.relpath("lib.tar")
    func.export_library(local_path)
    remote_path = '/home/firefly/Desktop/lib.tar'

    if mode == 'local_cpu':
        remote = rpc.LocalSession()
    else:
        # The following is my environment, change this to the IP address of your target device
        host = "10.158.176.86"
        port = 9090
        remote = rpc.connect(host, port)

    remote.upload(local_path, remote_path)
    func = remote.load_module(remote_path)

    # create arrays on the remote device
    if (mode == 'local_cpu' or mode == 'remote_cpu'):
        dev = remote.cpu()
    elif (mode == 'remote_opencl'):
        dev = remote.cl()
    a = tvm.nd.array(np.random.uniform(size=1024).astype(var_list[0].dtype), dev)
    b = tvm.nd.array(np.zeros(1024, dtype=var_list[0].dtype), dev)
    # the function will run on the remote device
    func(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    time_f = func.time_evaluator(func.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print("%g secs/op" % cost)