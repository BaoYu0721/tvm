#include <iostream>
#include <fstream>
#include <map>
#include <cmath>
#include "yaml-cpp/yaml.h"
#include <any>
#include <cassert>

#define INPUT_TYPE 0
#define GT_TYPE 1
#define OUTPUT_TYPE 2

void parseShapeTxt(std::string file_name, int& n, int& c, int& h, int& w)
{
    FILE *fp = fopen(file_name.c_str(), "r");
    fscanf(fp, "(%d, %d, %d, %d)", &n, &c, &h, &w);
    fclose(fp);
}

template<typename T>
void getDataFromBin(const std::string &file_name, T *data_ptr, int size)
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

void splitString(const std::string &tar_str, std::string seperator, std::vector<std::string> &split_res)
{
    std::size_t previous = 0;
    std::size_t current = tar_str.find(seperator);
    while (current != std::string::npos)
    {
        if (current > previous)
        {
            split_res.emplace_back(tar_str.substr(previous, current - previous));
        }
        previous = current + seperator.size();
        current = tar_str.find(seperator, previous);
    }
    if (previous != tar_str.size())
    {
        split_res.emplace_back(tar_str.substr(previous));
    }
}

std::string wildcardToSpecIdx(const std::string &fn_with_wc, int idx)
{
    std::vector<std::string> split_res;
    splitString(fn_with_wc, "*", split_res);
    assert(split_res.size() == 2 && "file name with wildcard error!\n");
    return split_res[0] + std::to_string(idx) + split_res[1];
}

template<typename TargetBufferT>
class BaseForwarder
{
protected:
    std::string mod_name;
    std::string target;
    int data_idx;
    std::vector<TargetBufferT> in_data_list;
    std::vector<TargetBufferT> out_data_list;
    std::vector<TargetBufferT> gt_data_list;
    std::vector<std::string> in_name_list;
    std::vector<std::string> out_name_list;
    std::vector<std::string> in_type_list;
    std::vector<std::string> out_type_list;
    std::vector<int> out_size_list;
    std::vector<int> in_size_list;
    YAML::Node cfg;

public:
    BaseForwarder(YAML::Node cfg);

    // buffer_type_flag: 0表示 input, 1 表示 ground_truth, 2 表示 output
    virtual void initTargetBuf(int n, int c, int h, int w, const std::string &type_str,
        int buffer_idx, int buffer_type_flag, TargetBufferT &res_buf) = 0;
    virtual void forwardOneTime() = 0;
    virtual void copyRawToTargetBuf(std::any raw_buf, TargetBufferT &tar_buf, int ele_size) = 0;
    virtual void copyTargetBufToRaw(TargetBufferT &tar_buf, std::any raw_buf, int ele_size) = 0;

    template<typename RawBufferT>
    void loadToSpecBuffer(const std::string &data_file, int buffer_idx, int in_or_gt);
    void loadToSpecBufferWrapper(const std::string &type_str, const std::string &data_file, int buffer_idx, int in_or_gt);
    void prepareInOutBuffer();

    template<typename RawBufferT>
    void checkOutAndGT(int buffer_idx);
    void checkAccuracy();
};

template<typename TargetBufferT>
BaseForwarder<TargetBufferT>::BaseForwarder(YAML::Node cfg)
{
    this->cfg = cfg;
    this->mod_name = cfg["model_name"].as<std::string>();
    target = cfg["target"].as<std::string>();
    data_idx = cfg["data_idx"].as<int>();

    in_name_list = cfg[mod_name]["in_tensor_names"].as<std::vector<std::string>>();
    out_name_list = cfg[mod_name]["out_tensor_names"].as<std::vector<std::string>>();
    in_type_list = cfg[mod_name]["in_types"].as<std::vector<std::string>>();
    out_type_list = cfg[mod_name]["out_types"].as<std::vector<std::string>>();
}

template<typename TargetBufferT>
template<typename RawBufferT>
void BaseForwarder<TargetBufferT>::loadToSpecBuffer(const std::string &data_file, int buffer_idx, int in_or_gt)
{
    // buffer_idx 指的是输入/输出 buffer 的 idx
    // in_or_gt: 为0表示这次加载的是 input，为1表示这次加载的是 groud truth
    int total_size = (in_or_gt == 0) ? in_size_list[buffer_idx] : out_size_list[buffer_idx];
    TargetBufferT *p_tar_buf = (in_or_gt == 0) ? &in_data_list[buffer_idx] : &gt_data_list[buffer_idx];
    RawBufferT *raw_buf = new RawBufferT[total_size];
    getDataFromBin(data_file, raw_buf, total_size);
    copyRawToTargetBuf(raw_buf, *p_tar_buf, total_size);
    delete [] raw_buf;
}

template<typename TargetBufferT>
void BaseForwarder<TargetBufferT>::loadToSpecBufferWrapper(const std::string &type_str, const std::string &data_file, int buffer_idx, int in_or_gt)
{
    if (type_str == "float32")
    {
        loadToSpecBuffer<float>(data_file, buffer_idx, in_or_gt);
    }
    else if (type_str == "int32")
    {
        loadToSpecBuffer<int>(data_file, buffer_idx, in_or_gt);
    }
    else if (type_str == "uint8")
    {
        loadToSpecBuffer<uint8_t>(data_file, buffer_idx, in_or_gt);
    }
}

template<typename TargetBufferT>
void BaseForwarder<TargetBufferT>::prepareInOutBuffer()
{
    int in, ic, ih, iw, on, oc, oh, ow;
    std::string data_dir = "../" + cfg[mod_name]["data_dir"].as<std::string>();
    std::vector<std::string> in_shape_names = cfg[mod_name]["in_shape_names"].as<std::vector<std::string>>();
    std::vector<std::string> out_shape_names = cfg[mod_name]["out_shape_names"].as<std::vector<std::string>>();
    std::vector<std::string> in_bin_names = cfg[mod_name]["in_bin_names"].as<std::vector<std::string>>();
    std::vector<std::string> out_bin_names = cfg[mod_name]["out_bin_names"].as<std::vector<std::string>>();
    for (int i = 0; i < in_type_list.size(); i++)
    {
        std::string in_shape_name = data_dir + wildcardToSpecIdx(in_shape_names[i], data_idx);
        std::string in_data_name = data_dir + wildcardToSpecIdx(in_bin_names[i], data_idx);
        parseShapeTxt(in_shape_name, in, ic, ih, iw);
        in_size_list.push_back(in * ic * ih * iw);
        TargetBufferT in_data;
        initTargetBuf(in, ic, ih, iw, in_type_list[i], i, INPUT_TYPE, in_data);
        in_data_list.emplace_back(in_data);
        loadToSpecBufferWrapper(in_type_list[i], in_data_name, i, INPUT_TYPE);
    }
    for (int i = 0; i < out_type_list.size(); i++)
    {
        std::string out_shape_name = data_dir + wildcardToSpecIdx(out_shape_names[i], data_idx);
        std::string out_data_name = data_dir + wildcardToSpecIdx(out_bin_names[i], data_idx);
        parseShapeTxt(out_shape_name, on, oc, oh, ow);
        out_size_list.push_back(on * oc * oh * ow);
        TargetBufferT out_data;
        TargetBufferT gt_data;
        initTargetBuf(on, oc, oh, ow, out_type_list[i], i, OUTPUT_TYPE, out_data);
        initTargetBuf(on, oc, oh, ow, out_type_list[i], i, GT_TYPE, gt_data);
        out_data_list.emplace_back(out_data);
        gt_data_list.emplace_back(gt_data);
        loadToSpecBufferWrapper(out_type_list[i], out_data_name, i, GT_TYPE);
    }
}

template<typename TargetBufferT>
template<typename RawBufferT>
void BaseForwarder<TargetBufferT>::checkOutAndGT(int buffer_idx)
{
    int total_out_size = out_size_list[buffer_idx];
    RawBufferT *out_raw = new RawBufferT[total_out_size];
    RawBufferT *gt_raw = new RawBufferT[total_out_size];
    copyTargetBufToRaw(out_data_list[buffer_idx], out_raw, total_out_size);
    copyTargetBufToRaw(gt_data_list[buffer_idx], gt_raw, total_out_size);
    checkDiff(out_raw, gt_raw, total_out_size);
    delete [] out_raw;
    delete [] gt_raw;
}

template<typename TargetBufferT>
void BaseForwarder<TargetBufferT>::checkAccuracy()
{
    for (int i = 0; i < out_type_list.size(); i++)
    {
        printf("The %d th output: ", i);
        if (out_type_list[i] == "float32")
        {
            checkOutAndGT<float>(i);
        }
        else if (out_type_list[i] == "int32")
        {
            checkOutAndGT<int>(i);
        }
        else if (out_type_list[i] == "uint8")
        {
            checkOutAndGT<uint8_t>(i);
        }
        printf("\n\n");
    }
}