#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <memory>
#include <map>
#include "common_tools.hpp"
#include <chrono>
#include <NvOnnxParser.h>

std::map<std::string, CodecNetProperty> name_prop_map;

class Logger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char* msg) noexcept override
    {
        std::cerr << int(severity) << ": " << msg << std::endl;
    }
};
static Logger gLogger;

std::string readBuffer(const std::string& path)
{
    std::string buffer;
    std::ifstream stream(path, std::ios::binary);

    if (stream)
    {
        stream >> std::noskipws;
        std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), std::back_inserter(buffer));
    }
    return buffer;
}

void writeBuffer(void* buffer, size_t size, const std::string& path)
{
    std::ofstream stream(path.c_str(), std::ios::binary);

    if (stream)
        stream.write(static_cast<char*>(buffer), size);
}

nvinfer1::ICudaEngine* createCudaEngine(const std::string &onnxModelPath)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cout << "ERROR: could not parse input engine." << std::endl;
        return nullptr;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    // if (builder->platformHasFastFp16())
    //     config->setFlag(BuilderFlag::kFP16);
    // builder->setMaxBatchSize(batchSize);

    auto profile = builder->createOptimizationProfile();
    // profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 176, 8, 12});
    // profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 176, 8, 12});
    // profile->setDimensions(network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 176, 8, 12});
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

class TrtEngineRunner
{
public:
    TrtEngineRunner(const std::string &engine_file_name, const std::string &model_name, const std::string &model_version);
    void prepareInOutBuffer(int data_idx);
    void forwardOneTime();
    void checkAccuracy();

private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;

    std::string mod_name;
    std::string mod_version;

    void *cuda_in_mem;
    void *cuda_out_mem;
    void *cuda_gt_mem;
    int total_in_size;
    int total_out_size;

    template<typename T>
    void allocBufferAndLoadData(const std::string &data_fn, int total_size, void *&cuda_mem);
    void allocBufferAndLoadDataAccordingToType(const std::string &data_fn, int total_size, const std::string &type_str, void *&cuda_mem);

    template<typename T>
    void checkOutAndGT();
};

TrtEngineRunner::TrtEngineRunner(const std::string &engine_file_name, const std::string &model_name, const std::string &model_version)
{
    this->mod_name = model_name;
    this->mod_version = model_version;
    std::string buffer = readBuffer(engine_file_name);
    if (buffer.size())
    {
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
        engine.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    }
    else
    {
        // std::string onnx_path = "../models/" + model_version + "_onnx/" + model_name + ".onnx";
        std::string onnx_path = "../tmp_trt_models/" + model_name + ".onnx";
        engine.reset(createCudaEngine(onnx_path));
        if (engine != nullptr)
        {
            std::unique_ptr<nvinfer1::IHostMemory> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), engine_file_name);
        }
    }

    context.reset(engine->createExecutionContext());
    cudaStreamCreate(&stream);

    cuda_in_mem = nullptr;
    cuda_out_mem = nullptr;
    cuda_gt_mem = nullptr;
}

template<typename T>
void TrtEngineRunner::allocBufferAndLoadData(const std::string &data_fn, int total_size, void *&cuda_mem)
{
    cudaMalloc(&cuda_mem, total_size * sizeof(T));
    if (data_fn != "")
    {
        T *data_ptr = new T[total_size];
        getDataFromBin(data_fn, data_ptr, total_size);
        cudaMemcpyAsync(cuda_mem, data_ptr, total_size * sizeof(T), cudaMemcpyHostToDevice, stream);
        delete [] data_ptr;
    }
}

void TrtEngineRunner::allocBufferAndLoadDataAccordingToType(const std::string &data_fn, int total_size, const std::string &type_str, void *&cuda_mem)
{
    if (type_str == "float32")
    {
        allocBufferAndLoadData<float>(data_fn, total_size, cuda_mem);
    }
    else if (type_str == "int32")
    {
        allocBufferAndLoadData<int32_t>(data_fn, total_size, cuda_mem);
    }
    else if (type_str == "uint8")
    {
        allocBufferAndLoadData<uint8_t>(data_fn, total_size, cuda_mem);
    }
}

void TrtEngineRunner::prepareInOutBuffer(int data_idx)
{
    int in, ic, ih, iw, on, oc, oh, ow;
    std::string data_dir = "./" + (name_prop_map.find(mod_name))->second.inout_data_dir + "/" + mod_version + "/";
    std::string in_shape_name = data_dir + "input_shape_" + std::to_string(data_idx) + ".txt";
    std::string out_shape_name = data_dir + "output_shape_" + std::to_string(data_idx) + ".txt";
    std::string in_data_name = data_dir + "input_" + std::to_string(data_idx) + ".bin";
    std::string out_data_name = data_dir + "output_" + std::to_string(data_idx) + ".bin";
    parseShapeTxt(in_shape_name, in, ic, ih, iw);
    parseShapeTxt(out_shape_name, on, oc, oh, ow);
    std::string in_type = (name_prop_map.find(mod_name))->second.in_type;
    std::string out_type = (name_prop_map.find(mod_name))->second.out_type;
    total_in_size = in * ic * ih * iw;
    total_out_size = on * oc * oh * ow;

    int32_t input_idx = engine->getBindingIndex((name_prop_map.find(mod_name))->second.input_name.c_str());
    nvinfer1::Dims4 input_dims = nvinfer1::Dims4{in, ic, ih, iw};
    context->setBindingDimensions(input_idx, input_dims);
    // int32_t output_idx = engine->getBindingIndex("z_decoder_out");
    // auto output_dims = context->getBindingDimensions(output_idx);

    allocBufferAndLoadDataAccordingToType(in_data_name, total_in_size, in_type, cuda_in_mem);
    allocBufferAndLoadDataAccordingToType("", total_out_size, out_type, cuda_out_mem);
    allocBufferAndLoadDataAccordingToType(out_data_name, total_out_size, out_type, cuda_gt_mem);
}

void TrtEngineRunner::forwardOneTime()
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed_time;

    void* bindings[] = {cuda_in_mem, cuda_out_mem};
    cudaEventRecord(start, stream);
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        std::cout << "trt forward error\n";
    }
    cudaEventRecord(end, stream);
    cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Inference spent: " <<  elapsed_time  << "ms\n";
}

template<typename T>
void TrtEngineRunner::checkOutAndGT()
{
    T *out_raw = new T[total_out_size];
    T *gt_raw = new T[total_out_size];
    cudaMemcpyAsync(out_raw, cuda_out_mem, total_out_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(gt_raw, cuda_gt_mem, total_out_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    checkDiff(out_raw, gt_raw, total_out_size);
    delete [] out_raw;
    delete [] gt_raw;
}

void TrtEngineRunner::checkAccuracy()
{
    std::string out_type = (name_prop_map.find(mod_name))->second.out_type;
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
    int data_idx = 0;
    parseSimpleCfg("../simple_cfg.txt", model_name, target_name, model_version, data_idx);
    prepareNamePropertyMap(model_version);
    // trtexec --onnx=./tmp_trt_models/z_decoder.onnx --workspace=64 --buildOnly --saveEngine=./tmp_trt_models/z_decoder.engine
    std::string engine_name = "../tmp_trt_models/" + model_name + ".engine";
    TrtEngineRunner trtr(engine_name, model_name, model_version);
    trtr.prepareInOutBuffer(data_idx);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    trtr.forwardOneTime();
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro secs\n\n";
    trtr.checkAccuracy();
    return 0;
}