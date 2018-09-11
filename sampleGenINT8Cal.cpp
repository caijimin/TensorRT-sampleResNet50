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

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

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
    printf("usage: %s -b mean.binaryproto \n"
                    "\t -d image-file-or-directory [-v validation-file]\n"
                    "\t [-h]\n\n", name);
}

vector <string> splitStr(const string& strSource, const string& strToken)
{
    vector <string> vecValues;
    if (strToken.empty()) {
        vecValues.push_back(strSource);
        return vecValues;
    }

    char *data = (char *)strSource.data();
    int size = strSource.size();
    char *pos = NULL;
    char *tok = (char *)strToken.data();
    int toklen = strToken.size();

    while( (pos = strstr(data, tok)) != NULL ) {
        vecValues.push_back(string(data, pos - data));
        size -= pos - data + toklen;
        data = pos + toklen;
    }

    if (size != 0)
        vecValues.push_back(string(data, size));
                                            
    return vecValues;
}

int main(int argc, char** argv)
{
    int c;
    char valfile[256];
    char outdir[256];
    
    while ((c = getopt(argc, argv, "b:d:v:o:h")) != -1) {
        switch (c) {
            case 'b':
                strcpy(mean, optarg);
                break;
            case 'd':
                strcpy(imageDir, optarg);
                break;
            case 'o':
                strcpy(outdir, optarg);
                break;
            case 'v':
                strcpy(valfile, optarg);
                break;
            case 'h':
                usage(argv[0]);
                return 0;
        }
    }

    if (strlen(outdir) < 1) {
        std::cout << "Output dir not specified\n";
        return -1;
    }
    if (strlen(valfile) < 1) {
        std::cout << "Validation file not specified\n";
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

    /** This vector stores paths to the processed images **/
    std::vector<float> labels;
    std::vector<float> batchData;
    char imagefile[256];

    // Parse mean file
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(mean);
    //printf("%d %d %d %d\n", meanBlob->getDimensions().n(), meanBlob->getDimensions().c(), meanBlob->getDimensions().h(),
    //        meanBlob->getDimensions().w());
    //    float pixelMean[3]{ 157.8806845, 163.71395787, 171.63139067 }; // also in BGR order
    parser->destroy();
    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

    // Subtract mean from image
    int batchSize = 50, batchNum=10;
    char line[256], outname[256];
    FILE *outfile;
    float *data = (float *)malloc(MAX_BATCHSIZE*INPUT_C * INPUT_H * INPUT_W *sizeof(float));;
    cv::Mat image;
    FILE *file = fopen(valfile, "r");
    if (!file)
        return -1;
    for (int i=0; i<batchNum; i++) {
        labels.clear();
        batchData.clear();
        sprintf(outname, "%s/batch%d", outdir, i);
        outfile = fopen(outname, "w");
        if (outfile == NULL)
            return -1;
        int s[4] = { batchSize, INPUT_C, INPUT_H, INPUT_W};
        fwrite(s, sizeof(int), 4, outfile);
        for (int j=0; j<batchSize; j++) {
            if (fgets(line, sizeof(line), file) == nullptr)
                return -1;;
            vector<string> tmp = splitStr(line, " ");
            sprintf(imagefile, "%s/%s", imageDir, tmp[0].c_str());
            labels.push_back(atoi(tmp[1].c_str()));
            printf("|%s|%f|\n", imagefile, labels[j]);
            image = cv::imread(imagefile, cv::IMREAD_COLOR);
            if (image.empty()) continue;
            cv::resize(image, image, cv::Size(INPUT_H,INPUT_W));
            for (int c = 0; c < INPUT_C; ++c) {
                // the color image to input should be in BGR order
                for (unsigned k = 0, volChl = INPUT_H*INPUT_W; k < volChl; ++k)
                    data[c*volChl + k] = float(image.data[k*INPUT_C + 2 - c]) - meanData[c*volChl + k];
            }
            fwrite(data, sizeof(float), INPUT_C * INPUT_H * INPUT_W, outfile);
        }
        for (int j=0; j<batchSize; j++) {
            float l = labels[j];
            fwrite(&l, sizeof(float), 1, outfile);
        }
        fclose(outfile);
    }

    fclose(file);
    return 0;

#if 0
                   

    fclose(file);
    readImagesArguments(imageNames, imageDir);
    if (imageNames.empty()) {
        cout << "No suitable images were found" <<endl;
        return -1;
    }
    for (unsigned int i=0; i<imageNames.size(); i++) {
        cout << imageNames[i] << endl;
    }


    batchSize = imageNames.size();
    if (batchSize > MAX_BATCHSIZE) {
        cout << "Max batch size is " << MAX_BATCHSIZE << ", will only handle first " << MAX_BATCHSIZE << " images" << endl;
        batchSize = MAX_BATCHSIZE;
    }
    // Create TRT model from caffe model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    caffeToTRTModel(prototxt, model, std::vector<std::string>{OUTPUT_BLOB_NAME}, batchSize, trtModelStream);
    assert(trtModelStream != nullptr);


    for (int i=0; i < batchSize; ++i) {
        image = cv::imread(imageNames[i], cv::IMREAD_COLOR);
        if (image.empty()) continue;
        cv::resize(image, image, cv::Size(INPUT_H,INPUT_W));

    }
    meanBlob->destroy();

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

    printf("Starting inference .............. \n");
    double total = 0.0;
    /** Start inference & calc performance **/
    for (int i = 0; i < iter; ++i) {
        auto t0 = Time::now();
        doInference(device, *context, data, prob, batchSize);
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        total += d.count();
    }


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    free(data);
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
#endif
    return  EXIT_SUCCESS;
}
