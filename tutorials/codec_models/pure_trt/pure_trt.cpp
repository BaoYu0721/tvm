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
#include <cstring>

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

class TrtEngineRunner : public BaseForwarder<void*>
{
public:
    TrtEngineRunner(YAML::Node cfg);
    void copyRawToTargetBuf(std::any raw_buf, void* &tar_buf, int ele_size);
    void copyTargetBufToRaw(void* &tar_buf, std::any raw_buf, int ele_size);
    void initTargetBuf(int n, int c, int h, int w, const std::string &type_str, 
        int buffer_idx, int buffer_type_flag, void* &res_buf);
    void forwardOneTime();
    ~TrtEngineRunner();

private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    std::vector<int> input_bindings;
    std::vector<int> output_bindings;
};

TrtEngineRunner::TrtEngineRunner(YAML::Node cfg) : BaseForwarder<void*>(cfg)
{
    std::string engine_file_name = "../tmp_trt_models/" + mod_name + ".engine";
    std::string buffer = readBuffer(engine_file_name);
    if (buffer.size())
    {
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
        engine.reset(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    }
    else
    {
        std::string onnx_path = "../tmp_trt_models/" + mod_name + ".onnx";
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
}

TrtEngineRunner::~TrtEngineRunner()
{
    for (size_t i = 0; i < in_data_list.size(); i++)
    {
        cudaFree(in_data_list[i]);
    }
    for (size_t i = 0; i < out_data_list.size(); i++)
    {
        cudaFree(out_data_list[i]);
        cudaFree(gt_data_list[i]);
    }
}

void TrtEngineRunner::initTargetBuf(int n, int c, int h, int w, const std::string &type_str,
    int buffer_idx, int buffer_type_flag, void* &res_buf)
{
    int total_size = n * c * h * w;
    res_buf = nullptr;
    if (type_str == "float32")
        cudaMalloc(&res_buf, total_size * sizeof(float));
    else if (type_str == "int32")
        cudaMalloc(&res_buf, total_size * sizeof(int));
    else if (type_str == "uint8")
        cudaMalloc(&res_buf, total_size * sizeof(uint8_t));

    if (buffer_type_flag == INPUT_TYPE)
    {
        int input_idx = engine->getBindingIndex(in_name_list[buffer_idx].c_str());
        nvinfer1::Dims4 input_dims = nvinfer1::Dims4{n, c, h, w};
        context->setBindingDimensions(input_idx, input_dims);
        input_bindings.push_back(input_idx);
    }
    else if (buffer_type_flag == OUTPUT_TYPE)
    {
        int out_idx = engine->getBindingIndex(out_name_list[buffer_idx].c_str());
        output_bindings.push_back(out_idx);
    }
}

void TrtEngineRunner::copyRawToTargetBuf(std::any raw_buf, void* &tar_buf, int ele_size)
{
    if (auto ptr = std::any_cast<float*>(&raw_buf))
        cudaMemcpyAsync(tar_buf, *ptr, ele_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    else if (auto ptr = std::any_cast<int*>(&raw_buf))
        cudaMemcpyAsync(tar_buf, *ptr, ele_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    else if (auto ptr = std::any_cast<uint8_t*>(&raw_buf))
        cudaMemcpyAsync(tar_buf, *ptr, ele_size * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
}

void TrtEngineRunner::copyTargetBufToRaw(void* &tar_buf, std::any raw_buf, int ele_size)
{
    if (auto ptr = std::any_cast<float*>(&raw_buf))
        cudaMemcpyAsync(*ptr, tar_buf, ele_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    else if (auto ptr = std::any_cast<int*>(&raw_buf))
        cudaMemcpyAsync(*ptr, tar_buf, ele_size * sizeof(int), cudaMemcpyDeviceToHost, stream);
    else if (auto ptr = std::any_cast<uint8_t*>(&raw_buf))
        cudaMemcpyAsync(*ptr, tar_buf, ele_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
}

void TrtEngineRunner::forwardOneTime()
{
    int binding_num = engine->getNbBindings();
    void **bindings = (void**)malloc(binding_num * sizeof(void*));
    memset(bindings, 0, binding_num * sizeof(void*));
    for (size_t i = 0; i < in_data_list.size(); i++)
    {
        bindings[input_bindings[i]] = in_data_list[i];
    }
    for (size_t i = 0; i < out_data_list.size(); i++)
    {
        bindings[output_bindings[i]] = out_data_list[i];
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed_time;
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
    free(bindings);
}

int main()
{
    YAML::Node cfg = YAML::LoadFile("../config.yaml");
    // trtexec --onnx=./tmp_trt_models/z_decoder.onnx --workspace=64 --buildOnly --saveEngine=./tmp_trt_models/z_decoder.engine
    TrtEngineRunner trtr(cfg);
    trtr.prepareInOutBuffer();
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    trtr.forwardOneTime();
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro secs\n\n";
    trtr.checkAccuracy();
    return 0;
}