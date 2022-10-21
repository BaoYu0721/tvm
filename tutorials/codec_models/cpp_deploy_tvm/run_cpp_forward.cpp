#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <iostream>
#include <map>
#include <fstream>
#include <cmath>
#include <chrono>

class CodecNetProperty
{
public:
    std::string inout_data_dir;
    std::string in_type;
    std::string out_type;
    std::string input_name;

    CodecNetProperty(const std::string data_dir, const std::string it, const std::string ot, const std::string iname) : inout_data_dir(data_dir),
        in_type(it), out_type(ot), input_name(iname)
    {

    }
};

void parseShapeTxt(std::string file_name, int& n, int& c, int& h, int& w)
{
    FILE *fp = fopen(file_name.c_str(), "r");
    fscanf(fp, "(%d, %d, %d, %d)", &n, &c, &h, &w);
    fclose(fp);
}

template<typename T>
void getDataFromBin(std::string file_name, T *data_ptr, int size)
{
    std::ifstream fin(file_name.c_str(), std::ifstream::binary);
    fin.read(reinterpret_cast<char*>(data_ptr), size * sizeof(T));
    fin.close();
}

template<typename T>
void checkDiff(T *d1, T *d2, int size)
{
    double max = -1e9, min = 1e9, sum = 0;
    for (int i = 0; i < size; i++)
    {
        double tmp = std::fabs(d1[i] - d2[i]);
        max = tmp > max ? tmp : max;
        min = tmp < min ? tmp : min;
        sum += tmp;
    }
    sum /= size;
    printf ("max_err: %lf, min_err: %lf, mean_err: %lf\n", max, min, sum);
}

class NetworkForward
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
    NetworkForward(const std::string mode_name, DLDeviceType dev_type);
    ~NetworkForward();

    template<typename IN_T, typename OUT_T>
    void loadInAndGTBuffer(const std::string in_data_name, const std::string out_data_name);

    void prepareInOutBuffer(const std::string &model_version, int data_idx);
    void forwardOneTime(DLDeviceType dev_type);

    template<typename T>
    void checkOutAndGT();

    void checkAccuracy();

    static std::map<std::string, CodecNetProperty> name_prop_map;
    static void prepareNamePropertyMap(const std::string &model_version);
};
std::map<std::string, CodecNetProperty> NetworkForward::name_prop_map;

NetworkForward::NetworkForward(const std::string mod_name, DLDeviceType dev_type)
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
    in_name = NetworkForward::name_prop_map.find(mod_name)->second.input_name;
    in_type = (NetworkForward::name_prop_map.find(mod_name))->second.in_type;
    out_type = (NetworkForward::name_prop_map.find(mod_name))->second.out_type;
    total_out_size = total_in_size = 0;
}

NetworkForward::~NetworkForward()
{
}

template<typename IN_T, typename OUT_T>
void NetworkForward::loadInAndGTBuffer(const std::string in_data_name, const std::string out_data_name)
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

void NetworkForward::prepareInOutBuffer(const std::string &model_version, int data_idx)
{
    int in, ic, ih, iw, on, oc, oh, ow;
    std::string data_dir = "./" + (NetworkForward::name_prop_map.find(mod_name))->second.inout_data_dir + "/" + model_version + "/";
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

void NetworkForward::forwardOneTime(DLDeviceType dev_type)
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
void NetworkForward::checkOutAndGT()
{
    T *out_raw = new T[total_out_size];
    T *gt_raw = new T[total_out_size];
    out_data.CopyToBytes(out_raw, total_out_size * sizeof(T));
    gt_data.CopyToBytes(gt_raw, total_out_size * sizeof(T));
    checkDiff(out_raw, gt_raw, total_out_size);
    delete [] out_raw;
    delete [] gt_raw;
}

void NetworkForward::checkAccuracy()
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

void NetworkForward::prepareNamePropertyMap(const std::string &model_version)
{
    std::string zd_in_type = "float32", zd_out_type = "float32";
    if (model_version == "u0.4.1")
    {
        zd_in_type = "int32";
        zd_out_type = "uint8";
    }
    name_prop_map.emplace(std::string("y_decoder"), CodecNetProperty("../test_data/ydecoder_inout", "float32", "float32", "y_decoder_data"));
    name_prop_map.emplace(std::string("y_encoder"), CodecNetProperty("../test_data/yencoder_inout", "float32", "float32", "y_encoder_data"));
    name_prop_map.emplace(std::string("z_decoder_int"), CodecNetProperty("../test_data/zdecoder_inout", zd_in_type, zd_out_type, "z_decoder_data"));
    name_prop_map.emplace(std::string("z_decoder"), CodecNetProperty("../test_data/zdecoder_inout", zd_in_type, zd_out_type, "z_decoder_data"));
    name_prop_map.emplace(std::string("z_encoder"), CodecNetProperty("../test_data/zencoder_inout", "float32", "float32", "z_encoder_data"));
}

void parseSimpleCfg(const std::string cfg_name, std::string &model_name, DLDeviceType &dev_type, std::string &model_version)
{
    std::ifstream fin(cfg_name.c_str());
    std::getline(fin, model_name);
    std::cout << model_name << "\n";
    std::string target_name;
    std::getline(fin, target_name);
    std::cout << target_name << "\n";
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
    std::getline(fin, model_version);
    std::cout << model_version << "\n";
    fin.close();
}

int main()
{
    std::string model_name, model_version;
    DLDeviceType dev_type;
    parseSimpleCfg("../simple_cfg.txt", model_name, dev_type, model_version);
    NetworkForward::prepareNamePropertyMap(model_version);
    // dev type: kDLCPU, kDLCUDA, etc.
    NetworkForward nf(model_name, dev_type);
    nf.prepareInOutBuffer(model_version, 0);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    nf.forwardOneTime(dev_type);
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro secs\n\n";
    nf.checkAccuracy();
    return 0;
}
