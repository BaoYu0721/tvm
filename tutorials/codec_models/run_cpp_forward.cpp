#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <iostream>
#include <map>
#include <fstream>
#include <cmath>

class CodecNetProperty
{
public:
    std::string inout_data_dir;
    std::string dtype;
    std::string input_name;

    CodecNetProperty(const std::string data_dir, const std::string dt, const std::string iname) : inout_data_dir(data_dir),
        dtype(dt), input_name(iname)
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
    std::string dtype;
    int total_out_size;
    int total_in_size;

public:
    NetworkForward(const std::string mode_name, DLDeviceType dev_type);
    ~NetworkForward();

    template<typename T>
    void loadInAndGTBuffer(const std::string in_data_name, const std::string out_data_name);

    void prepareInOutBuffer(int data_idx);
    void forwardOneTime();

    template<typename T>
    void checkOutAndGT();

    void checkAccuracy();

    static std::map<std::string, CodecNetProperty> name_prop_map;
    static void prepareNamePropertyMap();
};
std::map<std::string, CodecNetProperty> NetworkForward::name_prop_map;

NetworkForward::NetworkForward(const std::string mod_name, DLDeviceType dev_type)
{
    this->mod_name = mod_name;
    dev.device_type = dev_type;
    dev.device_id = 0;
    std::string dylib_path = "./libs/" + mod_name + ".so";
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(dylib_path);
    tvm::runtime::Module graph_exe_mod = mod_factory.GetFunction("default")(dev);
    set_input = graph_exe_mod.GetFunction("set_input");
    get_output = graph_exe_mod.GetFunction("get_output");
    run = graph_exe_mod.GetFunction("run");
    in_name = NetworkForward::name_prop_map.find(mod_name)->second.input_name;
    dtype = (NetworkForward::name_prop_map.find(mod_name))->second.dtype;
    total_out_size = total_in_size = 0;
}

NetworkForward::~NetworkForward()
{
}

template<typename T>
void NetworkForward::loadInAndGTBuffer(const std::string in_data_name, const std::string out_data_name)
{
    T *in_raw = new T[total_in_size];
    T *gt_raw = new T[total_out_size];
    getDataFromBin(in_data_name, in_raw, total_in_size);
    getDataFromBin(out_data_name, gt_raw, total_out_size);
    in_data.CopyFromBytes(in_raw, total_in_size * sizeof(T));
    gt_data.CopyFromBytes(gt_raw, total_out_size * sizeof(T));
    delete [] in_raw;
    delete [] gt_raw;
}

void NetworkForward::prepareInOutBuffer(int data_idx)
{
    int in, ic, ih, iw, on, oc, oh, ow;
    std::string data_dir = "./" + (NetworkForward::name_prop_map.find(mod_name))->second.inout_data_dir + "/";
    std::string in_shape_name = data_dir + "input_shape_" + std::to_string(data_idx) + ".txt";
    std::string out_shape_name = data_dir + "output_shape_" + std::to_string(data_idx) + ".txt";
    std::string in_data_name = data_dir + "input_" + std::to_string(data_idx) + ".bin";
    std::string out_data_name = data_dir + "output_" + std::to_string(data_idx) + ".bin";
    parseShapeTxt(in_shape_name, in, ic, ih, iw);
    parseShapeTxt(out_shape_name, on, oc, oh, ow);
    total_in_size = in * ic * ih * iw;
    total_out_size = on * oc * oh * ow;
    DLDataType ndarr_type = DLDataType{kDLFloat, 32, 1};
    if (dtype == "int32")
    {
        ndarr_type.code = kDLInt;
    }
    in_data = tvm::runtime::NDArray::Empty({in, ic, ih, iw}, ndarr_type, dev);
    out_data = tvm::runtime::NDArray::Empty({on, oc, oh, ow}, ndarr_type, dev);
    gt_data = tvm::runtime::NDArray::Empty({on, oc, oh, ow}, ndarr_type, dev);
    if (dtype == "float32")
    {
        loadInAndGTBuffer<float>(in_data_name, out_data_name);
    }
    else if (dtype == "int32")
    {
        loadInAndGTBuffer<int>(in_data_name, out_data_name);
    }
}

void NetworkForward::forwardOneTime()
{
    if (dtype == "float32" || dtype == "int32")
    {
        set_input(in_name, in_data);
        run();
        get_output(0, out_data);
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
    if (dtype == "float32")
    {
        checkOutAndGT<float>();
    }
    else if (dtype == "int32")
    {
        checkOutAndGT<int>();
    }
}

void NetworkForward::prepareNamePropertyMap()
{
    name_prop_map.emplace(std::string("y_decoder"), CodecNetProperty("ydecoder_inout", "float32", "y_decoder_data"));
    name_prop_map.emplace(std::string("y_encoder"), CodecNetProperty("yencoder_inout", "float32", "y_encoder_data"));
    name_prop_map.emplace(std::string("z_decoder_int"), CodecNetProperty("zdecoder_inout", "int32", "z_decoder_data"));
    name_prop_map.emplace(std::string("z_decoder"), CodecNetProperty("zdecoder_inout", "int32", "z_decoder_data"));
    name_prop_map.emplace(std::string("z_encoder"), CodecNetProperty("zencoder_inout", "float32", "z_encoder_data"));
}

void parseSimpleCfg(const std::string cfg_name, std::string &model_name, DLDeviceType &dev_type)
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
    fin.close();
}

int main()
{
    std::string model_name;
    DLDeviceType dev_type;
    parseSimpleCfg("./simple_cfg.txt", model_name, dev_type);
    NetworkForward::prepareNamePropertyMap();
    // dev type: kDLCPU, kDLCUDA, etc.
    NetworkForward nf(model_name, dev_type);
    nf.prepareInOutBuffer(0);
    nf.forwardOneTime();
    nf.checkAccuracy();
    return 0;
}
