#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <iostream>
#include <map>
#include <chrono>
#include "common_tools.hpp"

std::map<std::string, CodecNetProperty> name_prop_map;

class TVMNetworkForward
{
private:
    DLDevice dev;
    std::string mod_name;
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run;
    tvm::runtime::NDArray in_data;
    tvm::runtime::NDArray out_data;
    tvm::runtime::NDArray gt_data;
    std::string in_name;
    std::string in_type;
    std::string out_type;
    int total_out_size;
    int total_in_size;

public:
    TVMNetworkForward(const std::string mode_name, DLDeviceType dev_type);
    ~TVMNetworkForward();

    template<typename IN_T, typename OUT_T>
    void loadInAndGTBuffer(const std::string in_data_name, const std::string out_data_name);

    void prepareInOutBuffer(const std::string &model_version, int data_idx);
    void forwardOneTime(DLDeviceType dev_type);

    template<typename T>
    void checkOutAndGT();

    void checkAccuracy();
};

TVMNetworkForward::TVMNetworkForward(const std::string mod_name, DLDeviceType dev_type)
{
    this->mod_name = mod_name;
    dev.device_type = dev_type;
    dev.device_id = 0;
    std::string dylib_path = "../libs/" + mod_name + ".so";
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(dylib_path);
    tvm::runtime::Module graph_exe_mod = mod_factory.GetFunction("default")(dev);
    set_input = graph_exe_mod.GetFunction("set_input");
    get_output = graph_exe_mod.GetFunction("get_output");
    run = graph_exe_mod.GetFunction("run");
    in_name = name_prop_map.find(mod_name)->second.input_name;
    in_type = (name_prop_map.find(mod_name))->second.in_type;
    out_type = (name_prop_map.find(mod_name))->second.out_type;
    total_out_size = total_in_size = 0;
}

TVMNetworkForward::~TVMNetworkForward()
{
}

template<typename IN_T, typename OUT_T>
void TVMNetworkForward::loadInAndGTBuffer(const std::string in_data_name, const std::string out_data_name)
{
    IN_T *in_raw = new IN_T[total_in_size];
    OUT_T *gt_raw = new OUT_T[total_out_size];
    getDataFromBin(in_data_name, in_raw, total_in_size);
    getDataFromBin(out_data_name, gt_raw, total_out_size);
    in_data.CopyFromBytes(in_raw, total_in_size * sizeof(IN_T));
    gt_data.CopyFromBytes(gt_raw, total_out_size * sizeof(OUT_T));
    delete [] in_raw;
    delete [] gt_raw;
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

void TVMNetworkForward::prepareInOutBuffer(const std::string &model_version, int data_idx)
{
    int in, ic, ih, iw, on, oc, oh, ow;
    std::string data_dir = "./" + (name_prop_map.find(mod_name))->second.inout_data_dir + "/" + model_version + "/";
    std::string in_shape_name = data_dir + "input_shape_" + std::to_string(data_idx) + ".txt";
    std::string out_shape_name = data_dir + "output_shape_" + std::to_string(data_idx) + ".txt";
    std::string in_data_name = data_dir + "input_" + std::to_string(data_idx) + ".bin";
    std::string out_data_name = data_dir + "output_" + std::to_string(data_idx) + ".bin";
    parseShapeTxt(in_shape_name, in, ic, ih, iw);
    parseShapeTxt(out_shape_name, on, oc, oh, ow);
    total_in_size = in * ic * ih * iw;
    total_out_size = on * oc * oh * ow;
    DLDataType ndin_type, ndout_type;
    getDLDataTypeFromStr(in_type, ndin_type);
    getDLDataTypeFromStr(out_type, ndout_type);
    in_data = tvm::runtime::NDArray::Empty({in, ic, ih, iw}, ndin_type, dev);
    out_data = tvm::runtime::NDArray::Empty({on, oc, oh, ow}, ndout_type, dev);
    gt_data = tvm::runtime::NDArray::Empty({on, oc, oh, ow}, ndout_type, dev);
    if (in_type == "float32" && out_type == "float32")
    {
        loadInAndGTBuffer<float, float>(in_data_name, out_data_name);
    }
    else if (in_type == "int32" && out_type == "uint8")
    {
        loadInAndGTBuffer<int, uint8_t>(in_data_name, out_data_name);
    }
}

void TVMNetworkForward::forwardOneTime(DLDeviceType dev_type)
{
    if (in_type == "float32" || in_type == "int32")
    {
        set_input(in_name, in_data);
        run();
        get_output(0, out_data);
        // 仅仅是为了测速，上面的run和get_output都是异步的。
        // 如果不测速，其实不用加下面的同步函数。同步函数会自动在后面的 CopyToBytes 中调用
        TVMSynchronize(dev_type, 0, NULL);
    }
}

template<typename T>
void TVMNetworkForward::checkOutAndGT()
{
    T *out_raw = new T[total_out_size];
    T *gt_raw = new T[total_out_size];
    out_data.CopyToBytes(out_raw, total_out_size * sizeof(T));
    gt_data.CopyToBytes(gt_raw, total_out_size * sizeof(T));
    checkDiff(out_raw, gt_raw, total_out_size);
    delete [] out_raw;
    delete [] gt_raw;
}

void TVMNetworkForward::checkAccuracy()
{
    if (out_type == "float32")
    {
        checkOutAndGT<float>();
    }
    else if (out_type == "int32")
    {
        checkOutAndGT<int>();
    }
    else if (out_type == "uint8")
    {
        checkOutAndGT<uint8_t>();
    }
}

int main()
{
    std::string model_name, model_version, target_name;
    // dev type: kDLCPU, kDLCUDA, etc.
    DLDeviceType dev_type;
    int data_idx = 0;
    parseSimpleCfg("../simple_cfg.txt", model_name, target_name, model_version, data_idx);
    if (target_name == "llvm")
    {
        dev_type = kDLCPU;
    }
    else if (target_name == "cuda")
    {
        dev_type = kDLCUDA;
    }
    else
    {
        dev_type = kDLExtDev;
    }
    prepareNamePropertyMap(model_version);
    TVMNetworkForward nf(model_name, dev_type);
    nf.prepareInOutBuffer(model_version, data_idx);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    nf.forwardOneTime(dev_type);
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro secs\n\n";
    nf.checkAccuracy();
    return 0;
}
