# TensorRT-sampleResNet50
NVIDIA TensorRT™ is a high-performance deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. It provides some samples, but can't read jpg files directly. This sample can read jpg files using opencv and measure the performance. Further more it can use INT8 to accelerate inference after calibration.

Change of Makefile
==================

Need download ans install TensorRT and opencv, suppose TensorRT sample is installed in `/workspace/tensorrt/samples` directory, `Makefile.config` need add opencv flags
```
- COMMON_LD_FLAGS += $(LIBPATHS) -L$(OUTDIR)
+ COMMON_LD_FLAGS += $(LIBPATHS) -L$(OUTDIR) -lopencv_core -lopencv_highgui -lopencv_imgproc
```
Also in `Makefile` add change `samples=... sampleResNet50 ... `


Usage
=====
```
usage: ../../bin/sample_resnet50 -m model_file -p prototxt_file -b mean.binaryproto 
	 -d image-file-or-directory [-n iteration]
	 -c Calibrate-directory [-v (validation)] 
	 [-e device] [-t FLOAT|HALF|INT8] [-h]
```
For FLOAT32 and FLOAT16, calibration is not required, Caffe model file, prototxt file and image mean file are necesssary. It will read image files from the specified directory, the baytch size is the image number in that directory. Do inference and measure the performance. Please see below examples.
```
/workspace/tensorrt/samples/sampleResNet50# ../../bin/sample_resnet50 -m ResNet-50-model_iter_500000.caffemodel -p ResNet-50-imagedata-deploy.prototxt -b data/mean.binaryproto -d /data/batchsize/2 -n 10
[ INFO ] Add file  1_58a542f5N9afc8f1e.jpg_90.jpg from directory /data/batchsize/2.
[ INFO ] Add file  1_10193529281_4.jpg_90.jpg from directory /data/batchsize/2.
/data/batchsize/2/1_58a54_90.jpg
/data/batchsize/2/1_1019.jpg
Reading Caffe prototxt: ResNet-50-imagedata-deploy.prototxt
Reading Caffe model: ResNet-50-model_iter_500000.caffemodel
Starting inference .............. 

Output:

File: /data/batchsize/2/1_58a54_90.jpg
0:  0.00388638
1: ********** 0.996114

File: /data/batchsize/2/1_1019.jpg
0:  1.01029e-08
1: ********** 1


Average running time of one iteration: 8.25702 ms

Average running time of one forward: 7.14863 ms
batchSize: 2, Throughput 242.218 fps

```
You should change
```
static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 2;
```
according to your needs.

INT8 support
============
Using INT8 during inference need calibration, `sampleGenINT8Cal.cpp` will generare calibration data set in `$OUDIR/batches/` directory, file name is `batch%d`.
```
../../bin/sample_resnet50 -m ResNet-50-model_iter_500000.caffemodel -p ResNet-50-imagedata-deploy.prototxt -b /data/mean.binaryproto -d /data/val -n 1000 -c /data/ -t INT8 -v
 
Total validation images: 1305, errors = 109, error rate = 8.35249%
```
