#include <iostream>
#include <fstream>
#include <map>
#include <cmath>

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

extern std::map<std::string, CodecNetProperty> name_prop_map;

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

void parseSimpleCfg(const std::string cfg_name, std::string &model_name, std::string &target_name, std::string &model_version, int &data_idx)
{
    std::ifstream fin(cfg_name.c_str());
    std::getline(fin, model_name);
    std::cout << model_name << "\n";
    std::getline(fin, target_name);
    std::cout << target_name << "\n";
    std::getline(fin, model_version);
    std::cout << model_version << "\n";
    std::string data_idx_str;
    std::getline(fin, data_idx_str);
    std::cout << data_idx_str << "\n";
    data_idx = std::stoi(data_idx_str);
    fin.close();
}

void prepareNamePropertyMap(const std::string &model_version)
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
