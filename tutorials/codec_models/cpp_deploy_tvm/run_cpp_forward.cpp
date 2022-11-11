#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "common_tools.hpp"
#include <typeinfo>

class TVMNetworkForward : public BaseForwarder<tvm::runtime::NDArray>
{
private:
    DLDevice dev;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run;

public:
    TVMNetworkForward(YAML::Node cfg);

    void copyRawToTargetBuf(std::any raw_buf, tvm::runtime::NDArray &tar_buf, int ele_size);
    void copyTargetBufToRaw(tvm::runtime::NDArray &tar_buf, std::any raw_buf, int ele_size);
    void initTargetBuf(int n, int c, int h, int w, const std::string &type_str,
        int buffer_idx, int buffer_type_flag, tvm::runtime::NDArray &res_buf);
    void forwardOneTime();
};

void TVMNetworkForward::copyRawToTargetBuf(std::any raw_buf, tvm::runtime::NDArray &tar_buf, int ele_size)
{
    if (auto ptr = std::any_cast<float*>(&raw_buf))
        tar_buf.CopyFromBytes(*ptr, ele_size * sizeof(float));
    else if (auto ptr = std::any_cast<int*>(&raw_buf))
        tar_buf.CopyFromBytes(*ptr, ele_size * sizeof(int));
    else if (auto ptr = std::any_cast<uint8_t*>(&raw_buf))
        tar_buf.CopyFromBytes(*ptr, ele_size * sizeof(uint8_t));
}

void TVMNetworkForward::copyTargetBufToRaw(tvm::runtime::NDArray &tar_buf, std::any raw_buf, int ele_size)
{
    if (auto ptr = std::any_cast<float*>(&raw_buf))
        tar_buf.CopyToBytes(*ptr, ele_size * sizeof(float));
    else if (auto ptr = std::any_cast<int*>(&raw_buf))
        tar_buf.CopyToBytes(*ptr, ele_size * sizeof(int));
    else if (auto ptr = std::any_cast<uint8_t*>(&raw_buf))
        tar_buf.CopyToBytes(*ptr, ele_size * sizeof(uint8_t));
}

TVMNetworkForward::TVMNetworkForward(YAML::Node cfg) : BaseForwarder<tvm::runtime::NDArray>(cfg)
{
    // dev type: kDLCPU, kDLCUDA, etc.
    if (target == "llvm")
    {
        dev.device_type = kDLCPU;
    }
    else if (target == "cuda")
    {
        dev.device_type = kDLCUDA;
    }
    else
    {
        dev.device_type = kDLExtDev;
    }
    dev.device_id = 0;

    std::string dylib_path = "../libs/" + mod_name + ".so";
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(dylib_path);
    tvm::runtime::Module graph_exe_mod = mod_factory.GetFunction("default")(dev);
    set_input = graph_exe_mod.GetFunction("set_input");
    get_output = graph_exe_mod.GetFunction("get_output");
    run = graph_exe_mod.GetFunction("run");
}

void getDLDataTypeFromStr(const std::string &type_str, DLDataType &type)
{
    type = DLDataType{kDLFloat, 32, 1};
    if (type_str == "int32")
    {
        type = DLDataType{kDLInt, 32, 1};
    }
    else if (type_str == "uint8")
    {
        type = DLDataType{kDLUInt, 8, 1};
    }
}

void TVMNetworkForward::initTargetBuf(int n, int c, int h, int w, const std::string &type_str,
    int buffer_idx, int buffer_type_flag, tvm::runtime::NDArray &res_buf)
{
    DLDataType dtype;
    getDLDataTypeFromStr(type_str, dtype);
    res_buf = tvm::runtime::NDArray::Empty({n, c, h, w}, dtype, dev);
}

void TVMNetworkForward::forwardOneTime()
{
    for (int i = 0; i < in_type_list.size(); i++)
    {
        set_input(in_name_list[i], in_data_list[i]);
    }
    run();
    for (int i = 0; i < out_type_list.size(); i++)
    {
        get_output(i, out_data_list[i]);
    }
    // 仅仅是为了测速，上面的run和get_output都是异步的。
    // 如果不测速，其实不用加下面的同步函数。同步函数会自动在后面的 CopyToBytes 中调用
    TVMSynchronize(dev.device_type, 0, NULL);
}

int main()
{
    YAML::Node cfg = YAML::LoadFile("../config.yaml");
    TVMNetworkForward nf(cfg);
    nf.prepareInOutBuffer();
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    nf.forwardOneTime();
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro secs\n\n";
    nf.checkAccuracy();
    return 0;
}
