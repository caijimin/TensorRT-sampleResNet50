//! This sample builds a TensorRT engine by importing a trained MNIST Caffe model.
//! It uses the engine to run inference on an input image of a digit.

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <dirent.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "BatchStream.h"
#include "LegacyCalibrator.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
const char* gNetworkName = "resnet-50";

static Logger gLogger;

// Attributes of MNIST Caffe model
static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 2;
static const int MAX_BATCHSIZE = 512;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

int batchsize = 1;

float forwardtime = 0.0;
char model[256];
char prototxt[256];
char mean[256];
char imageDir[256];
int iter = 1;

char caliDir[256];
DataType modelDataType = DataType::kFLOAT;

std::string locateFile(const std::string& input)
{
    std::string s = std::string(caliDir) + "/" + input;
    return s;
}

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
	Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
		: mStream(stream), mReadCache(readCache)
	{
		DimsNCHW dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
		CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		mStream.reset(firstBatch);
	}

	virtual ~Int8EntropyCalibrator()
	{
		CHECK(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (!mStream.next())
			return false;

		CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		assert(!strcmp(names[0], INPUT_BLOB_NAME));
		bindings[0] = mDeviceInput;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		mCalibrationCache.clear();
		std::ifstream input(calibrationTableName(), std::ios::binary);
		input >> std::noskipws;
		if (mReadCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibrationTableName(), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        return std::string("CalibrationTable") + gNetworkName;
    }
	BatchStream mStream;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};

void caffeToTRTModel(const std::string& deployFile,           // Path of Caffe prototxt file
                     const std::string& modelFile,            // Path of Caffe model file
                     const std::vector<std::string>& outputs, // Names of network outputs
                     unsigned int maxBatchSize,               // Note: Must be at least as large as the batch we want to run with
                     DataType dataType,
                     IInt8Calibrator* calibrator,
                     IHostMemory*& trtModelStream)            // Output buffer for the TRT model
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse caffe model to populate network, then set the outputs
    const std::string deployFpath = prototxt;
    const std::string modelFpath = model;
    std::cout << "Reading Caffe prototxt: " << deployFpath << "\n";
    std::cout << "Reading Caffe model: " << modelFpath << "\n";
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFpath.c_str(),
                                                              modelFpath.c_str(),
                                                              *network,
                                                              dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);

    // Specify output tensors of network
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(10 << 20);

    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);

    builder->setInt8Mode(dataType == DataType::kINT8);
    builder->setFp16Mode(dataType == DataType::kHALF);
    if (dataType == DataType::kINT8)
        builder->setInt8Calibrator(calibrator);

    // Build engine
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // Destroy parser and network
    network->destroy();
    parser->destroy();

    // Serialize engine and destroy it
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();

    shutdownProtobufLibrary();
}

void doInference(int device, IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end; //calculate run time
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    float ms;
    cudaEventRecord(start, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    //context.execute(batchSize, buffers);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    forwardtime += ms;
    
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

/**
  * @brief This function check input args and find images in given folder
  */
void readImagesArguments(std::vector<std::string> &images, const std::string& arg) 
{
    struct stat sb;
    if (stat(arg.c_str(), &sb) != 0) {
        std::cout << "[ WARNING ] File " << arg << " cannot be opened!" << std::endl;
        return;
    }

    if (S_ISDIR(sb.st_mode)) {
        DIR *dp;
        dp = opendir(arg.c_str());
        if (dp == nullptr) {
            std::cout << "[ WARNING ] Directory " << arg << " cannot be opened!" << std::endl;
            return;
        }

        struct dirent *ep;
        while (nullptr != (ep = readdir(dp))) {
            std::string fileName = ep->d_name;
            if (fileName == "." || fileName == "..") continue;
            //else if (fileName.find(".jpg") == string::npos && fileName.find(".bmp") == string::npos) continue;
            std::cout << "[ INFO ] Add file  " << ep->d_name << " from directory " << arg << "." << std::endl;
            images.push_back(arg + "/" + ep->d_name);
        }
    } else {
        images.push_back(arg);
    }
}

void usage(char *name)
{
    printf("usage: %s -m model_file -p prototxt_file -b mean.binaryproto \n"
                    "\t -d image-file-or-directory [-n iteration]\n"
                    "\t -c Calibrate-directory [-v (validation)] \n"
                    "\t [-e device] [-t FLOAT|HALF|INT8] [-h]\n\n", name);
}


int main(int argc, char** argv)
{
    int c;
    int batchSize = 1;
    int device = 0;
    bool validation = false; // if "-v" is set, validate dataset
    
    while ((c = getopt(argc, argv, "m:p:b:d:n:t:e:c:vh")) != -1) {
        switch (c) {
            case 'm':
                strcpy(model, optarg);
                break;
            case 'p':
                strcpy(prototxt, optarg);
                break;
            case 'b':
                strcpy(mean, optarg);
                break;
            case 'd':
                strcpy(imageDir, optarg);
                break;
            case 'c':
                strcpy(caliDir, optarg);
                break;
            case 'v':
                validation = true;
                break;
            case 'e':
                device = atoi(optarg);
                cudaSetDevice(device);
                break;
            case 't':
                if (strstr(optarg, "HALF") != nullptr)
                    modelDataType = DataType::kHALF;
                else if (strstr(optarg, "INT8") != nullptr)
                    modelDataType = DataType::kINT8;
                break;
            case 'n':
                iter = atoi(optarg);
                if (iter == 0)
                    iter = 1;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
        }
    }

    if (strlen(model) < 1) {
        std::cout << "model file not specified\n";
        return -1;
    }
    if (strlen(prototxt) < 1) {
        std::cout << "prototxt file not specified\n";
        return -1;
    }
    if (strlen(mean) < 1) {
        std::cout << "mean.binarytproto file not specified\n";
        return -1;
    }
    if (strlen(imageDir) < 1) {
        std::cout << "Image file or directory not specified\n";
        return -1;
    }

    if ( modelDataType == DataType::kINT8 && strlen(caliDir) < 1) {
        std::cout << "Need calibration for INT8, calibration directory not specified\n";
        return -1;
    }
    /** This vector stores paths to the processed images **/
    std::vector<std::string> imageNames;
    std::vector<int> labels;
    readImagesArguments(imageNames, imageDir);
    if (imageNames.empty()) {
        cout << "No suitable images were found" <<endl;
        return -1;
    }
    for (unsigned int i=0; i<imageNames.size(); i++) {
        if (validation) {
            char tmp[256];
            strcpy(tmp, imageNames[i].c_str());
            char *bname = basename(tmp);
            labels.push_back(atoi(bname));
            cout << imageNames[i] << " : " << labels[i] << endl;
        } else {
            cout << imageNames[i] <<  endl;
        }
    }

    cv::Mat image;

    batchSize = imageNames.size();
    if (batchSize > MAX_BATCHSIZE) {
        cout << "Max batch size is " << MAX_BATCHSIZE << ", will only handle first " << MAX_BATCHSIZE << " images" << endl;
        batchSize = MAX_BATCHSIZE;
    }

    if (validation)
        batchSize = 1;

    // Create TRT model from caffe model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    if (modelDataType == DataType::kINT8) {
        BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
        Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH);
        caffeToTRTModel(prototxt, model, std::vector<std::string>{OUTPUT_BLOB_NAME}, batchSize, 
            modelDataType, &calibrator, trtModelStream);
    } else {
        caffeToTRTModel(prototxt, model, std::vector<std::string>{OUTPUT_BLOB_NAME}, batchSize, 
            modelDataType, nullptr, trtModelStream);
    }
    assert(trtModelStream != nullptr);

    // Parse mean file
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(mean);
    //printf("%d %d %d %d\n", meanBlob->getDimensions().n(), meanBlob->getDimensions().c(), meanBlob->getDimensions().h(),
    //        meanBlob->getDimensions().w());
    //    float pixelMean[3]{ 157.8806845, 163.71395787, 171.63139067 }; // also in BGR order
    parser->destroy();


    // Deserialize engine we serialized earlier
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference on input data
    float prob[MAX_BATCHSIZE*OUTPUT_SIZE];
 
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;

    // Subtract mean from image
    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

    float *data = (float *)malloc(MAX_BATCHSIZE*INPUT_C * INPUT_H * INPUT_W *sizeof(float));;
    double total = 0.0;
    if (validation) {
        int errors = 0;
        printf("Starting validation .............. \n");
        for (unsigned int i=0; i<imageNames.size(); ++i) {
            image = cv::imread(imageNames[i], cv::IMREAD_COLOR);
            if (image.empty()) continue;
            cv::resize(image, image, cv::Size(INPUT_H,INPUT_W));

            for (int c = 0; c < INPUT_C; ++c) {
                // the color image to input should be in BGR order
                for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
                    data[c*volChl + j] = float(image.data[j*INPUT_C + 2 - c]) - meanData[c*volChl + j];
            }
            doInference(device, *context, data, prob, batchSize);
            float val{0.0f};
            int idx{0};
            for (unsigned int k = 0; k < OUTPUT_SIZE; k++) {
                val = std::max(val, prob[k]);
                if (val == prob[k]) idx = k;
            }
            if (idx != labels[i]) {
                errors++;
                cout << imageNames[i] << "validation fail, label: " << labels[i] << ", idx: " << idx << ", val: " << val <<endl;
            }
        }
        std::cout << endl <<  "Total validation images: " << imageNames.size() << ", errors = " << errors 
            << ", error rate = " << (float)errors*100/imageNames.size() << "%" << std::endl;
    } else {
        for (int i=0; i < batchSize; ++i) {
            image = cv::imread(imageNames[i], cv::IMREAD_COLOR);
            if (image.empty()) continue;
            cv::resize(image, image, cv::Size(INPUT_H,INPUT_W));

            for (int c = 0; c < INPUT_C; ++c) {
                // the color image to input should be in BGR order
                for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
                    data[i*INPUT_C * INPUT_H * INPUT_W + c*volChl + j] = 
                        float(image.data[j*INPUT_C + 2 - c]) - meanData[c*volChl + j]; //pixelMean[c];
            }
        }
        printf("Starting inference .............. \n");
        /** Start inference & calc performance **/
        for (int i = 0; i < iter; ++i) {
            auto t0 = Time::now();
            doInference(device, *context, data, prob, batchSize);
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }
        // Print histogram of the output distribution
        std::cout << "\nOutput:\n\n";
        for (int n=0; n < batchSize; ++n) {
            cout << "File: " << imageNames[n] << endl;
            for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
            {
                std::cout << i << ": " << std::string(int(std::floor(prob[n*OUTPUT_SIZE+i] * 10 + 0.5f)), '*') 
                    << " " << prob[n*OUTPUT_SIZE+i]<< "\n";
            }
            std::cout << std::endl;
        }

        /** Show performance results **/
        double infertime = total / iter;
        std::cout << endl <<  "Average running time of one iteration: " << infertime << " ms" << std::endl;
        std::cout << endl <<  "Average running time of one forward: " << forwardtime/iter << " ms" << std::endl;
        std::cout << "batchSize: " << batchSize << ", Throughput " << 1000/infertime*batchSize << " fps" << std::endl;
    }

    meanBlob->destroy();

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    free(data);

    return  EXIT_SUCCESS;
}
