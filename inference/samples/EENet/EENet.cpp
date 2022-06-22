#include <cuda_runtime_api.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <numeric>

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "NvInfer.h"

using samplesCommon::SampleUniquePtr;
typedef std::chrono::high_resolution_clock Clock;

const std::string gSampleName = "TensorRT.B_AlexnetRedesigned_CIFAR10";
        
struct EENetParams : public samplesCommon::SampleParams {
    int inputC;                  // The input channel
    int inputH;                  // The input height
    int inputW;                  // The input width
    int outputSize;              // The output size
    std::string weightsFile;     // The filename of the weights file
    std::string testFile;        // The filename of the test set file
    std::string labelFile;       // The filename of the test label file
};

class EENet {
    public:
        EENet(const EENetParams& params) : mParams(params) {}

        bool build();                         // Build the network engines
        bool prepare();                       // Prepare engines for test
        bool test(int sample_total);          // Runs the TensorRt inference engine for sample
        bool teardown();                      // Clean up

    private:
        EENetParams mParams;                                                 // Parameters for the sample
        std::map<std::string, nvinfer1::Weights> mWeightMap;                 // The weight name
        std::vector<std::unique_ptr<samplesCommon::HostMemory>> weightsMem;  // Host weights memory holder
    
        void loadTestSample(int num);
        int infer(float threashold);
        void bb_thread();
        bool validateOutput();
        bool check_exit_criteria(float threshold); 
        std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file); 
        
        int currentLabel;
        bool terminate = false; 

        bool buildEngine0(
            const SampleUniquePtr<nvinfer1::IBuilder>& builder,
            const SampleUniquePtr<nvinfer1::IRuntime>& runtime,
            cudaStream_t profileStream
        ); 
        SampleUniquePtr<nvinfer1::ICudaEngine> mEngine0{nullptr};
        SampleUniquePtr<nvinfer1::IExecutionContext>mContext0{nullptr}; 

        bool buildEngine1(
            const SampleUniquePtr<nvinfer1::IBuilder>& builder,
            const SampleUniquePtr<nvinfer1::IRuntime>& runtime,
            cudaStream_t profileStream
        ); 
        SampleUniquePtr<nvinfer1::ICudaEngine> mEngine1{nullptr};
        SampleUniquePtr<nvinfer1::IExecutionContext>mContext1{nullptr}; 

        bool buildEngine2(
            const SampleUniquePtr<nvinfer1::IBuilder>& builder,
            const SampleUniquePtr<nvinfer1::IRuntime>& runtime,
            cudaStream_t profileStream
        ); 
        SampleUniquePtr<nvinfer1::ICudaEngine> mEngine2{nullptr};
        SampleUniquePtr<nvinfer1::IExecutionContext>mContext2{nullptr}; 
        void *input, *output;
        void *binding0[3];
        void *binding1[3];
        void *binding2[2];
        nvinfer1::Dims mIntermediateDims0;
        nvinfer1::Dims mIntermediateDims1;

        int accuracy[3] = {0, 0, 0};
        float latency[3] = {0, 0, 0};
        int outputs[3] = {0, 0, 0};

        std::vector<std::vector<float>> latencies = {{}, {}, {}};

        template <typename T>
        SampleUniquePtr<T> makeUnique(T* t) {
            return SampleUniquePtr<T>{t};
        }
}; 

bool EENet::build() {
    mWeightMap = loadWeights(locateFile(mParams.weightsFile, mParams.dataDirs));

    auto builder = makeUnique(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        sample::gLogError << "Create inference builder failed." << std::endl;
        return false;
    }

    auto runtime = makeUnique(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime) {
        sample::gLogError << "Runtime object creation failed." << std::endl;
        return false;
    }

    try {
        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream) {
            return false;
        }
        bool result; 
        result = buildEngine0(builder, runtime, *profileStream)
              && buildEngine1(builder, runtime, *profileStream)
              && buildEngine2(builder, runtime, *profileStream);
        return result;
    } catch (std::runtime_error& e) {
        sample::gLogError << e.what() << std::endl;
        return false;
    }
}       

bool EENet::prepare() {
    mContext0 = makeUnique(mEngine0->createExecutionContext());
    if (!mContext0) {
        sample::gLogError << "Context0 build failed." << std::endl;
        return false; 
    }

    mContext1 = makeUnique(mEngine1->createExecutionContext());
    if (!mContext1) {
        sample::gLogError << "Context1 build failed." << std::endl;
        return false; 
    }

    mContext2 = makeUnique(mEngine2->createExecutionContext());
    if (!mContext2) {
        sample::gLogError << "Context2 build failed." << std::endl;
        return false; 
    }

    CHECK(cudaMallocHost(&input, mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));
    CHECK(cudaMallocHost(&output, mParams.outputSize * sizeof(float)));
    CHECK(cudaMalloc(&binding0[0], mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));
    CHECK(cudaMalloc(&binding0[1], samplesCommon::volume(mIntermediateDims0) * sizeof(float)));
    CHECK(cudaMalloc(&binding0[2], mParams.outputSize * sizeof(float)));
    binding1[0] = binding0[1];
    CHECK(cudaMalloc(&binding1[1], samplesCommon::volume(mIntermediateDims1) * sizeof(float)));
    binding1[2] = binding0[2];
    binding2[0] = binding1[1];
    binding2[1] = binding0[2];

    return true;
}

bool EENet::test(int sample_total) {
    auto start_time = Clock::now();
    auto end_time = Clock::now();
    int EE;
    float exe_time; 

    // warm up
    for (int i = 0; i < 2000; i++) {
        loadTestSample(i);
        infer(i);
    }   
        

    std::cout << "\n\nPerforming Experiments\n";
    
        std::ofstream latfile;
    latfile.open("lat.txt", std::ios_base::app);
        
    for (int i = 0; i < sample_total; i++) {
        loadTestSample(i);

        start_time = Clock::now();
        EE = infer(i);
        end_time = Clock::now();
        
        exe_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()/1000000.0;
        

        outputs[EE]++;
        latencies[EE].push_back(exe_time);
        accuracy[EE] += validateOutput();
        
        latfile << exe_time << std::endl;
    }
    
    latfile.close();
    
    for (unsigned int i = 0; i < latencies.size(); i++) {
        std::sort(latencies[i].begin(), latencies[i].end());
        latency[i] = std::accumulate(latencies[i].begin() + latencies[i].size() * 0, latencies[i].begin() + latencies[i].size() * 0.5, latency[i]);
        latency[i] /= int(latencies[i].size() * 0.5) - int(latencies[i].size() * 0);
    }
        
    float totalLatency = 0; totalLatency = std::accumulate(latency, latency+3, totalLatency);
    int totalAccuracy = 0; totalAccuracy = std::accumulate(accuracy, accuracy+3, totalAccuracy);
    std::cout << "///////////////////////////////////////////////////////////\nSummary:\n"
              << "Total outputs: " << sample_total
              << "    Exit0: " << outputs[0] * 100.0 / sample_total << "%"
              << "    Exit1: " << outputs[1] * 100.0 / sample_total << "%"
              << "    Exit2: " << outputs[2] * 100.0 / sample_total << "%"
              << "\nAverage Latency:  " << totalLatency / 3 << "ms"
              << "    Exit0: " << latency[0] << "ms"
              << "    Exit1: " << latency[1] << "ms"
              << "    Exit2: " << latency[2] << "ms"
              << "\nAverage Accuracy: " << totalAccuracy * 100.0 / sample_total << "%"
              << "    Exit0: " << accuracy[0] * 100.0 / outputs[0] << "%"
              << "    Exit1: " << accuracy[1] * 100.0 / outputs[1] << "%"
              << "    Exit2: " << accuracy[2] * 100.0 / outputs[2] << "%"
              << std::endl;

    std::ofstream datafile;
    datafile.open("data.txt", std::ios_base::app);
    datafile << totalLatency / 3 << ", ";
    datafile << latency[0] << ", ";
    datafile << latency[1] << ", ";
    datafile << latency[2] << ", ";

    datafile << std::endl;
    datafile.close();
    return true;
}       
int EENet::infer(float threshold) {
    CHECK(cudaMemcpy(binding0[0], input, mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float), cudaMemcpyHostToDevice));

    if (!mContext0->executeV2(binding0)) { std::cout << "Execute engine0 error" << std::endl; }
    CHECK(cudaMemcpy(output, binding0[2], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    if (!check_exit_criteria((bool)(((int)threshold+0)%3))) {
        if (!mContext1->executeV2(binding1)) { std::cout << "Execute engine1 error" << std::endl; }
        CHECK(cudaMemcpy(output, binding1[2], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));

        if (!check_exit_criteria((bool)(((int)threshold+1)%3))) {
            if (!mContext2->executeV2(binding2)) { std::cout << "Execute engine2 error" << std::endl; }
            CHECK(cudaMemcpy(output, binding2[1], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));

            return 2;
        }
        return 1;
    }
    return 0;
}

bool EENet::check_exit_criteria(float threshold) {
    const float* bufRaw = static_cast<const float*>(output);
    std::vector<float> prob(bufRaw, bufRaw + mParams.outputSize * sizeof(float));
    return prob[std::max_element(prob.begin(), prob.end()) - prob.begin()] > threshold;
}
        

bool EENet::validateOutput() {
    const float* bufRaw = static_cast<const float*>(output);
    std::vector<float> prob(bufRaw, bufRaw + mParams.outputSize * sizeof(float));
    int predictedDigit = std::max_element(prob.begin(), prob.end()) - prob.begin();
    

    return currentLabel == predictedDigit;
}       
void EENet::loadTestSample(int num) {

    std::ifstream testFile(locateFile(mParams.testFile, mParams.dataDirs), std::ifstream::binary);
    ASSERT(testFile.is_open() && "Attempting to read from a file that is not open."); 

    Dims4 inputDims{1, mParams.inputC, mParams.inputH, mParams.inputW};
    size_t vol = samplesCommon::volume(inputDims);
    std::vector<uint8_t> label(1);
    
    // load label
    testFile.seekg(0 + 3073 * num);
    testFile.read(reinterpret_cast<char*>(label.data()), 1);
    currentLabel = (int)label[0];

    //load data
    std::vector<uint8_t> fileData(vol);
    testFile.seekg(1 + 3073 * num);
    testFile.read(reinterpret_cast<char*>(fileData.data()), vol);

    // Normalize and copy to the host buffer.
    float* hostDataBuffer = static_cast<float*>(input);
    std::transform(fileData.begin(), fileData.end(), hostDataBuffer,
        [](uint8_t x) { return static_cast<float>(((x / 255.0) - 0.4733630120754242) / 0.2515689432621002); });
    return;
}      
bool EENet::teardown() {
    CHECK(cudaFreeHost(input));
    CHECK(cudaFreeHost(output));
    CHECK(cudaFree(binding0[0]));
    CHECK(cudaFree(binding0[1]));
    CHECK(cudaFree(binding0[2]));
    CHECK(cudaFree(binding1[1]));

    return true;
}

std::map<std::string, nvinfer1::Weights> EENet::loadWeights(const std::string& file) {
    sample::gLogInfo << "Loading weights: " << file << std::endl;

    // Open weights file
    std::ifstream input(file, std::ios::binary);
    ASSERT(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    ASSERT(count > 0 && "Invalid weight map file.");

    std::map<std::string, nvinfer1::Weights> weightMap;
    while (count--) {
        nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
        int type;
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);

        // Load blob
        if (wt.type == DataType::kFLOAT) {
            // Use uint32_t to create host memory to avoid additional conversion.
            auto mem = new samplesCommon::TypedHostMemory<uint32_t, nvinfer1::DataType::kFLOAT>(size);
            weightsMem.emplace_back(mem);
            uint32_t* val = mem->raw();
            for (uint32_t x = 0; x < size; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF) {
            // HalfMemory's raw type is uint16_t
            auto mem = new samplesCommon::HalfMemory(size);
            weightsMem.emplace_back(mem);
            auto val = mem->raw();
            for (uint32_t x = 0; x < size; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}       

EENetParams initializeSampleParams() {
    EENetParams params;
    params.dataDirs.push_back("data/");
    params.inputC = 3;
    params.inputH = 32;
    params.inputW = 32;
    params.outputSize = 10;
    params.weightsFile = "B_AlexnetRedesigned_CIFAR10.wts";
    params.testFile = "CIFAR10/cifar-10-batches-bin/test_batch.bin";
    params.labelFile = "CIFAR10/cifar-10-batches-bin/test_batch.bin";
    return params;
}

int main(int argc, char** argv) {
    auto Test = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(Test);
    EENet sample{initializeSampleParams()};

    if (!sample.build()) {
        return sample::gLogger.reportFail(Test);
    }
    if (!sample.prepare()) {
        return sample::gLogger.reportFail(Test);
    }
    if (!sample.test(10000)) {
        return sample::gLogger.reportFail(Test);
    }

    if (!sample.teardown()) {
        return sample::gLogger.reportFail(Test);
    }

    return sample::gLogger.reportPass(Test);
} 

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + ".shortcut.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".shortcut.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *relu2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *relu2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}
            

bool EENet::buildEngine0(
    const SampleUniquePtr<nvinfer1::IBuilder>& builder,
    const SampleUniquePtr<nvinfer1::IRuntime>& runtime, 
    cudaStream_t profileStream) {

    INetworkDefinition* Network0 = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!Network0) {
        sample::gLogError << "Create network failed." << std::endl;
        return false;
    }

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    ITensor* data = Network0->addInput(
        "Input",
        DataType::kFLOAT,
        Dims4{1, mParams.inputC, mParams.inputH, mParams.inputW}
    );
    ASSERT(data);


    // Convolution layer with 32 outputs, 5x5 filter, 1x1 stride, 2x2 padding
    IConvolutionLayer* bb_Conv2d_0 = Network0->addConvolutionNd( 
        *data, 
        32, 
        Dims{2, {5, 5}}, 
        mWeightMap["backbone.0.0.weight"], 
        mWeightMap["backbone.0.0.bias"] 
    );
    ASSERT(bb_Conv2d_0);
    bb_Conv2d_0->setStrideNd(Dims{2, {1, 1}});
    bb_Conv2d_0->setPaddingNd(Dims{2, {2, 2}});
    bb_Conv2d_0->setName("bb_Conv2d_0");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_1 = Network0->addActivation(
        *bb_Conv2d_0->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_1);
    bb_ReLU_1->setName("bb_ReLU_1");

    // Max pooling layer with kernel of 3x3 and stride size of 2x2
    IPoolingLayer* bb_MaxPool2d_2 = Network0->addPoolingNd(
        *bb_ReLU_1->getOutput(0),
        PoolingType::kMAX,
        Dims{2, {3, 3}}
    );
    ASSERT(bb_MaxPool2d_2);
    bb_MaxPool2d_2->setStrideNd(Dims{2, {2, 2}});
    bb_MaxPool2d_2->setName("bb_MaxPool2d_2");
    bb_MaxPool2d_2->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    // Local Response Normalization layer with alpha = 0.000050, beta = 0.75 and k = 1.0
    ILRNLayer* bb_LocalResponseNorm_3 = Network0->addLRN( 
        *bb_MaxPool2d_2->getOutput(0), 
        3, 
        0.000050, 
        0.75, 
        1.0 
    ); 
    ASSERT(bb_LocalResponseNorm_3);
    bb_LocalResponseNorm_3->setName("bb_LocalResponseNorm_3");


    // Convolution layer with 32 outputs, 5x5 filter, 1x1 stride, 2x2 padding
    IConvolutionLayer* ee0_Conv2d_0 = Network0->addConvolutionNd( 
        *bb_LocalResponseNorm_3->getOutput(0), 
        32, 
        Dims{2, {5, 5}}, 
        mWeightMap["exits.0.0.weight"], 
        mWeightMap["exits.0.0.bias"] 
    );
    ASSERT(ee0_Conv2d_0);
    ee0_Conv2d_0->setStrideNd(Dims{2, {1, 1}});
    ee0_Conv2d_0->setPaddingNd(Dims{2, {2, 2}});
    ee0_Conv2d_0->setName("ee0_Conv2d_0");

    // ReLU activation layer
    IActivationLayer* ee0_ReLU_1 = Network0->addActivation(
        *ee0_Conv2d_0->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(ee0_ReLU_1);
    ee0_ReLU_1->setName("ee0_ReLU_1");

    // Max pooling layer with kernel of 3x3 and stride size of 2x2
    IPoolingLayer* ee0_MaxPool2d_2 = Network0->addPoolingNd(
        *ee0_ReLU_1->getOutput(0),
        PoolingType::kMAX,
        Dims{2, {3, 3}}
    );
    ASSERT(ee0_MaxPool2d_2);
    ee0_MaxPool2d_2->setStrideNd(Dims{2, {2, 2}});
    ee0_MaxPool2d_2->setName("ee0_MaxPool2d_2");
    ee0_MaxPool2d_2->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    // Convolution layer with 32 outputs, 3x3 filter, 1x1 stride, 1x1 padding
    IConvolutionLayer* ee0_Conv2d_3 = Network0->addConvolutionNd( 
        *ee0_MaxPool2d_2->getOutput(0), 
        32, 
        Dims{2, {3, 3}}, 
        mWeightMap["exits.0.3.weight"], 
        mWeightMap["exits.0.3.bias"] 
    );
    ASSERT(ee0_Conv2d_3);
    ee0_Conv2d_3->setStrideNd(Dims{2, {1, 1}});
    ee0_Conv2d_3->setPaddingNd(Dims{2, {1, 1}});
    ee0_Conv2d_3->setName("ee0_Conv2d_3");

    // ReLU activation layer
    IActivationLayer* ee0_ReLU_4 = Network0->addActivation(
        *ee0_Conv2d_3->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(ee0_ReLU_4);
    ee0_ReLU_4->setName("ee0_ReLU_4");

    // Max pooling layer with kernel of 3x3 and stride size of 2x2
    IPoolingLayer* ee0_MaxPool2d_5 = Network0->addPoolingNd(
        *ee0_ReLU_4->getOutput(0),
        PoolingType::kMAX,
        Dims{2, {3, 3}}
    );
    ASSERT(ee0_MaxPool2d_5);
    ee0_MaxPool2d_5->setStrideNd(Dims{2, {2, 2}});
    ee0_MaxPool2d_5->setName("ee0_MaxPool2d_5");
    ee0_MaxPool2d_5->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    // Flatten layer flattens t0 512 features
    IShuffleLayer* ee0_Flatten_6 = Network0->addShuffle( 
        *ee0_MaxPool2d_5->getOutput(0) 
    ); 
    ASSERT(ee0_Flatten_6); 
    ee0_Flatten_6->setReshapeDimensions(Dims{2, {1, 512}});
    ee0_Flatten_6->setName("ee0_Flatten_6");

    // Linear layer with 512 input features and 10 output features
    IConstantLayer* ee0_Linear_7_weights = Network0->addConstant( 
        Dims{2, {10, 512}}, 
        mWeightMap["exits.0.7.weight"] 
    ); 
    ASSERT(ee0_Linear_7_weights); 
    ee0_Linear_7_weights->setName("ee0_Linear_7_weights");
    IMatrixMultiplyLayer* ee0_Linear_7 = Network0->addMatrixMultiply(
        *ee0_Flatten_6->getOutput(0), 
        MatrixOperation::kNONE, 
        *ee0_Linear_7_weights->getOutput(0), 
        MatrixOperation::kTRANSPOSE
    ); 
    ASSERT(ee0_Linear_7); 
    ee0_Linear_7->setName("ee0_Linear_7");

    // Softmax Layer
    ISoftMaxLayer* ee0_Softmax_8 = Network0->addSoftMax( 
        *ee0_Linear_7->getOutput(0) 
    ); 
    ASSERT(ee0_Softmax_8);
    ee0_Softmax_8->setName("ee0_Softmax_8");
    ee0_Softmax_8->setAxes(2);

    Network0->markOutput(*bb_LocalResponseNorm_3->getOutput(0));
    mIntermediateDims0 = bb_LocalResponseNorm_3->getOutput(0)->getDimensions();

    Network0->markOutput(*ee0_Softmax_8->getOutput(0));


    auto Config0 = makeUnique(builder->createBuilderConfig());
    if (!Config0) {
        sample::gLogError << "Create builder config failed." << std::endl;
        return false;
    }
    Config0->setMaxWorkspaceSize(4096_MiB);
    Config0->setProfileStream(profileStream);

    SampleUniquePtr<nvinfer1::IHostMemory> Plan0 = makeUnique(
        builder->buildSerializedNetwork(*Network0, *Config0));
    if (!Plan0) {
        sample::gLogError << "serialized engine0 build failed." << std::endl;
        return false;
    }

    mEngine0 = makeUnique(
        runtime->deserializeCudaEngine(Plan0->data(), Plan0->size()));
    if (!mEngine0) {
        sample::gLogError << "engine0 deserialization failed." << std::endl;
        return false;
    }

    return true;
}


bool EENet::buildEngine1(
    const SampleUniquePtr<nvinfer1::IBuilder>& builder,
    const SampleUniquePtr<nvinfer1::IRuntime>& runtime, 
    cudaStream_t profileStream) {

    INetworkDefinition* Network1 = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!Network1) {
        sample::gLogError << "Create network failed." << std::endl;
        return false;
    }

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    ITensor* data = Network1->addInput(
        "Input",
        DataType::kFLOAT,
        mIntermediateDims0
    );
    ASSERT(data);


    // Convolution layer with 64 outputs, 5x5 filter, 1x1 stride, 2x2 padding
    IConvolutionLayer* bb_Conv2d_4 = Network1->addConvolutionNd( 
        *data, 
        64, 
        Dims{2, {5, 5}}, 
        mWeightMap["backbone.1.0.weight"], 
        mWeightMap["backbone.1.0.bias"] 
    );
    ASSERT(bb_Conv2d_4);
    bb_Conv2d_4->setStrideNd(Dims{2, {1, 1}});
    bb_Conv2d_4->setPaddingNd(Dims{2, {2, 2}});
    bb_Conv2d_4->setName("bb_Conv2d_4");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_5 = Network1->addActivation(
        *bb_Conv2d_4->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_5);
    bb_ReLU_5->setName("bb_ReLU_5");

    // Max pooling layer with kernel of 3x3 and stride size of 2x2
    IPoolingLayer* bb_MaxPool2d_6 = Network1->addPoolingNd(
        *bb_ReLU_5->getOutput(0),
        PoolingType::kMAX,
        Dims{2, {3, 3}}
    );
    ASSERT(bb_MaxPool2d_6);
    bb_MaxPool2d_6->setStrideNd(Dims{2, {2, 2}});
    bb_MaxPool2d_6->setName("bb_MaxPool2d_6");
    bb_MaxPool2d_6->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    // Local Response Normalization layer with alpha = 0.000050, beta = 0.75 and k = 1.0
    ILRNLayer* bb_LocalResponseNorm_7 = Network1->addLRN( 
        *bb_MaxPool2d_6->getOutput(0), 
        3, 
        0.000050, 
        0.75, 
        1.0 
    ); 
    ASSERT(bb_LocalResponseNorm_7);
    bb_LocalResponseNorm_7->setName("bb_LocalResponseNorm_7");


    // Convolution layer with 96 outputs, 3x3 filter, 1x1 stride, 1x1 padding
    IConvolutionLayer* bb_Conv2d_8 = Network1->addConvolutionNd( 
        *bb_LocalResponseNorm_7->getOutput(0), 
        96, 
        Dims{2, {3, 3}}, 
        mWeightMap["backbone.1.4.weight"], 
        mWeightMap["backbone.1.4.bias"] 
    );
    ASSERT(bb_Conv2d_8);
    bb_Conv2d_8->setStrideNd(Dims{2, {1, 1}});
    bb_Conv2d_8->setPaddingNd(Dims{2, {1, 1}});
    bb_Conv2d_8->setName("bb_Conv2d_8");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_9 = Network1->addActivation(
        *bb_Conv2d_8->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_9);
    bb_ReLU_9->setName("bb_ReLU_9");

    // Convolution layer with 32 outputs, 3x3 filter, 1x1 stride, 1x1 padding
    IConvolutionLayer* ee1_Conv2d_0 = Network1->addConvolutionNd( 
        *bb_ReLU_9->getOutput(0), 
        32, 
        Dims{2, {3, 3}}, 
        mWeightMap["exits.1.0.weight"], 
        mWeightMap["exits.1.0.bias"] 
    );
    ASSERT(ee1_Conv2d_0);
    ee1_Conv2d_0->setStrideNd(Dims{2, {1, 1}});
    ee1_Conv2d_0->setPaddingNd(Dims{2, {1, 1}});
    ee1_Conv2d_0->setName("ee1_Conv2d_0");

    // ReLU activation layer
    IActivationLayer* ee1_ReLU_1 = Network1->addActivation(
        *ee1_Conv2d_0->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(ee1_ReLU_1);
    ee1_ReLU_1->setName("ee1_ReLU_1");

    // Max pooling layer with kernel of 3x3 and stride size of 3x3
    IPoolingLayer* ee1_MaxPool2d_2 = Network1->addPoolingNd(
        *ee1_ReLU_1->getOutput(0),
        PoolingType::kMAX,
        Dims{2, {3, 3}}
    );
    ASSERT(ee1_MaxPool2d_2);
    ee1_MaxPool2d_2->setStrideNd(Dims{2, {3, 3}});
    ee1_MaxPool2d_2->setName("ee1_MaxPool2d_2");
    ee1_MaxPool2d_2->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    // Flatten layer flattens t0 288 features
    IShuffleLayer* ee1_Flatten_3 = Network1->addShuffle( 
        *ee1_MaxPool2d_2->getOutput(0) 
    ); 
    ASSERT(ee1_Flatten_3); 
    ee1_Flatten_3->setReshapeDimensions(Dims{2, {1, 288}});
    ee1_Flatten_3->setName("ee1_Flatten_3");

    // Linear layer with 288 input features and 10 output features
    IConstantLayer* ee1_Linear_4_weights = Network1->addConstant( 
        Dims{2, {10, 288}}, 
        mWeightMap["exits.1.4.weight"] 
    ); 
    ASSERT(ee1_Linear_4_weights); 
    ee1_Linear_4_weights->setName("ee1_Linear_4_weights");
    IMatrixMultiplyLayer* ee1_Linear_4 = Network1->addMatrixMultiply(
        *ee1_Flatten_3->getOutput(0), 
        MatrixOperation::kNONE, 
        *ee1_Linear_4_weights->getOutput(0), 
        MatrixOperation::kTRANSPOSE
    ); 
    ASSERT(ee1_Linear_4); 
    ee1_Linear_4->setName("ee1_Linear_4");

    // Softmax Layer
    ISoftMaxLayer* ee1_Softmax_5 = Network1->addSoftMax( 
        *ee1_Linear_4->getOutput(0) 
    ); 
    ASSERT(ee1_Softmax_5);
    ee1_Softmax_5->setName("ee1_Softmax_5");
    ee1_Softmax_5->setAxes(2);

    Network1->markOutput(*bb_ReLU_9->getOutput(0));
    mIntermediateDims1 = bb_ReLU_9->getOutput(0)->getDimensions();

    Network1->markOutput(*ee1_Softmax_5->getOutput(0));


    auto Config1 = makeUnique(builder->createBuilderConfig());
    if (!Config1) {
        sample::gLogError << "Create builder config failed." << std::endl;
        return false;
    }
    Config1->setMaxWorkspaceSize(4096_MiB);
    Config1->setProfileStream(profileStream);

    SampleUniquePtr<nvinfer1::IHostMemory> Plan1 = makeUnique(
        builder->buildSerializedNetwork(*Network1, *Config1));
    if (!Plan1) {
        sample::gLogError << "serialized engine1 build failed." << std::endl;
        return false;
    }

    mEngine1 = makeUnique(
        runtime->deserializeCudaEngine(Plan1->data(), Plan1->size()));
    if (!mEngine1) {
        sample::gLogError << "engine1 deserialization failed." << std::endl;
        return false;
    }

    return true;
}


bool EENet::buildEngine2(
    const SampleUniquePtr<nvinfer1::IBuilder>& builder,
    const SampleUniquePtr<nvinfer1::IRuntime>& runtime, 
    cudaStream_t profileStream) {

    INetworkDefinition* Network2 = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!Network2) {
        sample::gLogError << "Create network failed." << std::endl;
        return false;
    }

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    ITensor* data = Network2->addInput(
        "Input",
        DataType::kFLOAT,
        mIntermediateDims1
    );
    ASSERT(data);


    // Convolution layer with 96 outputs, 3x3 filter, 1x1 stride, 1x1 padding
    IConvolutionLayer* bb_Conv2d_10 = Network2->addConvolutionNd( 
        *data, 
        96, 
        Dims{2, {3, 3}}, 
        mWeightMap["backbone.2.0.weight"], 
        mWeightMap["backbone.2.0.bias"] 
    );
    ASSERT(bb_Conv2d_10);
    bb_Conv2d_10->setStrideNd(Dims{2, {1, 1}});
    bb_Conv2d_10->setPaddingNd(Dims{2, {1, 1}});
    bb_Conv2d_10->setName("bb_Conv2d_10");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_11 = Network2->addActivation(
        *bb_Conv2d_10->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_11);
    bb_ReLU_11->setName("bb_ReLU_11");

    // Convolution layer with 64 outputs, 3x3 filter, 1x1 stride, 1x1 padding
    IConvolutionLayer* bb_Conv2d_12 = Network2->addConvolutionNd( 
        *bb_ReLU_11->getOutput(0), 
        64, 
        Dims{2, {3, 3}}, 
        mWeightMap["backbone.2.2.weight"], 
        mWeightMap["backbone.2.2.bias"] 
    );
    ASSERT(bb_Conv2d_12);
    bb_Conv2d_12->setStrideNd(Dims{2, {1, 1}});
    bb_Conv2d_12->setPaddingNd(Dims{2, {1, 1}});
    bb_Conv2d_12->setName("bb_Conv2d_12");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_13 = Network2->addActivation(
        *bb_Conv2d_12->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_13);
    bb_ReLU_13->setName("bb_ReLU_13");

    // Max pooling layer with kernel of 3x3 and stride size of 2x2
    IPoolingLayer* bb_MaxPool2d_14 = Network2->addPoolingNd(
        *bb_ReLU_13->getOutput(0),
        PoolingType::kMAX,
        Dims{2, {3, 3}}
    );
    ASSERT(bb_MaxPool2d_14);
    bb_MaxPool2d_14->setStrideNd(Dims{2, {2, 2}});
    bb_MaxPool2d_14->setName("bb_MaxPool2d_14");
    bb_MaxPool2d_14->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    // Flatten layer flattens t0 1024 features
    IShuffleLayer* bb_Flatten_15 = Network2->addShuffle( 
        *bb_MaxPool2d_14->getOutput(0) 
    ); 
    ASSERT(bb_Flatten_15); 
    bb_Flatten_15->setReshapeDimensions(Dims{2, {1, 1024}});
    bb_Flatten_15->setName("bb_Flatten_15");

    // Linear layer with 1024 input features and 256 output features
    IConstantLayer* bb_Linear_16_weights = Network2->addConstant( 
        Dims{2, {256, 1024}}, 
        mWeightMap["backbone.2.6.weight"] 
    ); 
    ASSERT(bb_Linear_16_weights); 
    bb_Linear_16_weights->setName("bb_Linear_16_weights");
    IMatrixMultiplyLayer* bb_Linear_16_matmul = Network2->addMatrixMultiply(
        *bb_Flatten_15->getOutput(0), 
        MatrixOperation::kNONE, 
        *bb_Linear_16_weights->getOutput(0), 
        MatrixOperation::kTRANSPOSE
    ); 
    ASSERT(bb_Linear_16_matmul); 
    bb_Linear_16_matmul->setName("bb_Linear_16_matmul");
    IConstantLayer* bb_Linear_16_bias = Network2->addConstant(
        Dims{2, {1, 256}}, 
        mWeightMap["backbone.2.6.bias"]
    ); 
    ASSERT(bb_Linear_16_bias); 
    bb_Linear_16_bias->setName("bb_Linear_16_bias");
    IElementWiseLayer* bb_Linear_16 = Network2->addElementWise(
        *bb_Linear_16_matmul->getOutput(0), 
        *bb_Linear_16_bias->getOutput(0), 
        ElementWiseOperation::kSUM
    ); 
    ASSERT(bb_Linear_16); 
    bb_Linear_16->setName("bb_Linear_16");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_17 = Network2->addActivation(
        *bb_Linear_16->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_17);
    bb_ReLU_17->setName("bb_ReLU_17");

    // Linear layer with 256 input features and 128 output features
    IConstantLayer* bb_Linear_18_weights = Network2->addConstant( 
        Dims{2, {128, 256}}, 
        mWeightMap["backbone.2.9.weight"] 
    ); 
    ASSERT(bb_Linear_18_weights); 
    bb_Linear_18_weights->setName("bb_Linear_18_weights");
    IMatrixMultiplyLayer* bb_Linear_18_matmul = Network2->addMatrixMultiply(
        *bb_ReLU_17->getOutput(0), 
        MatrixOperation::kNONE, 
        *bb_Linear_18_weights->getOutput(0), 
        MatrixOperation::kTRANSPOSE
    ); 
    ASSERT(bb_Linear_18_matmul); 
    bb_Linear_18_matmul->setName("bb_Linear_18_matmul");
    IConstantLayer* bb_Linear_18_bias = Network2->addConstant(
        Dims{2, {1, 128}}, 
        mWeightMap["backbone.2.9.bias"]
    ); 
    ASSERT(bb_Linear_18_bias); 
    bb_Linear_18_bias->setName("bb_Linear_18_bias");
    IElementWiseLayer* bb_Linear_18 = Network2->addElementWise(
        *bb_Linear_18_matmul->getOutput(0), 
        *bb_Linear_18_bias->getOutput(0), 
        ElementWiseOperation::kSUM
    ); 
    ASSERT(bb_Linear_18); 
    bb_Linear_18->setName("bb_Linear_18");

    // ReLU activation layer
    IActivationLayer* bb_ReLU_19 = Network2->addActivation(
        *bb_Linear_18->getOutput(0),
        ActivationType::kRELU
    );
    ASSERT(bb_ReLU_19);
    bb_ReLU_19->setName("bb_ReLU_19");

    // Linear layer with 128 input features and 10 output features
    IConstantLayer* bb_Linear_20_weights = Network2->addConstant( 
        Dims{2, {10, 128}}, 
        mWeightMap["exits.2.0.weight"] 
    ); 
    ASSERT(bb_Linear_20_weights); 
    bb_Linear_20_weights->setName("bb_Linear_20_weights");
    IMatrixMultiplyLayer* bb_Linear_20 = Network2->addMatrixMultiply(
        *bb_ReLU_19->getOutput(0), 
        MatrixOperation::kNONE, 
        *bb_Linear_20_weights->getOutput(0), 
        MatrixOperation::kTRANSPOSE
    ); 
    ASSERT(bb_Linear_20); 
    bb_Linear_20->setName("bb_Linear_20");

    // Softmax Layer
    ISoftMaxLayer* bb_Softmax_21 = Network2->addSoftMax( 
        *bb_Linear_20->getOutput(0) 
    ); 
    ASSERT(bb_Softmax_21);
    bb_Softmax_21->setName("bb_Softmax_21");
    bb_Softmax_21->setAxes(2);

    Network2->markOutput(*bb_Softmax_21->getOutput(0));


    auto Config2 = makeUnique(builder->createBuilderConfig());
    if (!Config2) {
        sample::gLogError << "Create builder config failed." << std::endl;
        return false;
    }
    Config2->setMaxWorkspaceSize(4096_MiB);
    Config2->setProfileStream(profileStream);

    SampleUniquePtr<nvinfer1::IHostMemory> Plan2 = makeUnique(
        builder->buildSerializedNetwork(*Network2, *Config2));
    if (!Plan2) {
        sample::gLogError << "serialized engine2 build failed." << std::endl;
        return false;
    }

    mEngine2 = makeUnique(
        runtime->deserializeCudaEngine(Plan2->data(), Plan2->size()));
    if (!mEngine2) {
        sample::gLogError << "engine2 deserialization failed." << std::endl;
        return false;
    }

    return true;
}

