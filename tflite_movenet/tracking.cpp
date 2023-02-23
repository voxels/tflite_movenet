#include "tracking.h"
#include <iostream>

#include <array>
#include <fstream>
#include <vector>
#include "bitmap_helpers.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/core/public/session.h"

/*
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye' : 1,
    'right_eye' : 2,
    'left_ear' : 3,
    'right_ear' : 4,
    'left_shoulder' : 5,
    'right_shoulder' : 6,
    'left_elbow' : 7,
    'right_elbow' : 8,
    'left_wrist' : 9,
    'right_wrist' : 10,
    'left_hip' : 11,
    'right_hip' : 12,
    'left_knee' : 13,
    'right_knee' : 14,
    'left_ankle' : 15,
    'right_ankle' : 16
}
*/

namespace movenet_tracking {


    void RunInference(Settings* settings) {
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

        if (!settings->model_name.c_str()) {
            std::cout << "no model file name" << std::endl;
            exit(-1);
        }

        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        model = tflite::FlatBufferModel::BuildFromFile(settings->model_name.c_str());
        if (!model) {
            std::cout << "Failed to map model " << settings->model_name << std::endl;
            exit(-1);
        }

        // Create an Interpreter with an InterpreterBuilder.
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            std::cout << "Failed to construct interpreter" << std::endl;;
            exit(-1);
        }

        interpreter->SetAllowFp16PrecisionForFp32(settings->allow_fp16);


        if (settings->number_of_threads != -1) {
            interpreter->SetNumThreads(settings->number_of_threads);
        }

        int image_width = 256;
        int image_height = 256;
        int image_channels = 3;

        std::vector<uint8_t> in = tflite::read_bmp(settings->input_jpg_name, &image_width,
            &image_height, &image_channels, settings);

        int input = interpreter->inputs()[0];
        if (settings->verbose) std::cout << "inputindex: " << input << "\n";


        int outputIndex = interpreter->outputs()[0];
        if (settings->verbose) std::cout << "outputindex: " << outputIndex << "\n";

        const std::vector<int> inputs = interpreter->inputs();
        const std::vector<int> outputs = interpreter->outputs();

        if (settings->verbose) {
            std::cout << "number of inputs: " << inputs.size() << "\n";
            std::cout << "number of outputs: " << outputs.size() << "\n";
        }

        TfLiteTensor* inputTensor = interpreter->tensor(interpreter->inputs()[0]);
        auto inputDataName = inputTensor->name;
        auto inputDataType = inputTensor->type;
        auto inputDataDimsSize = inputTensor->dims->size;
        auto inputDataAllocationType = inputTensor->allocation_type;
        // std::cout << inputDataName << "\ttype: " << inputDataType << "\tsize: " << inputDataDimsSize << "\tbytes: " << inputTensor->bytes << "\tallocation type: " << inputDataAllocationType << std::endl;

        interpreter->ResizeInputTensor(input, { 1, image_width, image_height, 3 });

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            printf("Error allocating tensors\n");
            // Return failure.
        }
        else {
            printf("Interpreter allocated tensors\n");
        }

        for (int ii = 0; ii < inputDataDimsSize; ii++) {
            std::cout << "resized dim " << ii << " " << "\t" << inputTensor->dims->data[ii] << std::endl;
        }

        uint8_t* typedInputTensor = interpreter->typed_input_tensor<uint8_t>(input);
        for (int ii = 0; ii < in.size(); ii++) {
            typedInputTensor[ii] = in[ii];
        }

        for (int i = 0; i < settings->loop_count; i++) {
            if (interpreter->Invoke() != kTfLiteOk) {
                std::cout << "Failed to invoke tflite!";
                exit(-1);
            }
        }

        TfLiteTensor* outputTensor = interpreter->tensor(interpreter->outputs()[0]);
        auto outputDataName = outputTensor->name;
        auto outputDataType = outputTensor->type;
        auto outputDataDimsSize = outputTensor->dims->size;
        auto outputDataAllocationType = outputTensor->allocation_type;
        //std::cout << outputDataName << "\ttype: " << outputDataType << "\tsize: " << outputDataDimsSize << "\tbytes: " << outputTensor->bytes << "\tallocation type: " << outputDataAllocationType << std::endl;
        for (int i = 0; i < outputTensor->dims->size; i++) {
            int data = outputTensor->dims->data[i];
            // std::cout << "Output dimensions data index\t" << i << "\tdata: " << data << "\n";
        }

        std::vector<float> unraveledArray;

        int output = interpreter->outputs()[0];
        TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];
        switch (interpreter->tensor(output)->type) {
        case kTfLiteFloat32:
            for (int i = 0; i < outputTensor->dims->data[0] * outputTensor->dims->data[1] * outputTensor->dims->data[2]; i++) {
                float data = interpreter->typed_output_tensor<float>(0)[i];
                if( settings->verbose) std::cout << "float raw data\t" << i << "\t" << data << "\n";
                unraveledArray.push_back(data);
            }
            break;
        case kTfLiteInt8:
            break;
        case kTfLiteUInt8:
            break;
        default:
            std::cout << "cannot handle output type "
                << interpreter->tensor(output)->type << " yet";
            exit(-1);
        }


        std::vector<std::vector<std::vector<float>>> reshapedArray;

        for (int j = 0; j < outputTensor->dims->data[1]; j++) {
            std::vector<std::vector< float >> personData;
            std::vector<float> jointData;
            for (int k = 0; k < outputTensor->dims->data[2]; k++) {
                int index = j * outputTensor->dims->data[2] + k;                
                if ( k < outputTensor->dims->data[2] - 5) {
                    float data = unraveledArray[index];
                    int innerIndex = k % 3;
                    if (innerIndex == 0) {
                        //std::cout << "joint values: " << jointData.size() << std::endl;
                        if (jointData.size() > 0) {
                            std::vector<float> jointCoordinatesAndScore;
                            for (int i = 0; i < jointData.size(); i++) {
                                jointCoordinatesAndScore.push_back(jointData[i]);
                            }
                            personData.push_back(jointCoordinatesAndScore);
                            jointData.clear();
                            jointData.push_back(data);
                            //std::cout << " clear vector index:\t" << index << " " << j << ":" << k << "\t" << data << "\t" << innerIndex << " " << jointCoordinatesAndScore.size() << "\n";
                        }
                        else {
                            jointData.push_back(data);
                            //std::cout << "build vector index:\t" << index << " " << j << ":" << k << "\t" << data << "\t" << innerIndex << "\n";
                        }
                    }
                    else {
                        jointData.push_back(data);
                        //std::cout << "add vector index:\t" << index << " " << j << ":" << k << "\t" << data << "\t" << innerIndex << "\n";
                        if (k == outputTensor->dims->data[2] - 6) {
                            std::vector<float> jointCoordinatesAndScore;
                            for (int i = 0; i < jointData.size(); i++) {
                                jointCoordinatesAndScore.push_back(jointData[i]);
                            }
                            personData.push_back(jointCoordinatesAndScore);
                        }
                    }
                    //std::cout << "person joints: " << personData.size() << std::endl;
                }
            }
            std::vector<std::vector< float >> foundPersonData;
            for (int ii = 0; ii < personData.size(); ii++) {
                foundPersonData.push_back(personData[ii]);
            }

            reshapedArray.push_back(foundPersonData);
            personData.clear();
        }

        std::cout << "Reshaped array size: (number of people)" << reshapedArray.size() << std::endl;

        for (int ii = 0; ii < 1 /*reshapedArray.size()*/; ii++) {
            std::vector<std::vector<float>> personData = reshapedArray[ii];
            std::cout << "Person Data size: (number of joints) " << personData.size() << std::endl;

            for (int jj = 0; jj < personData.size(); jj++) {
                std::vector<float> jointData = personData[jj];
                //std::cout << "Joint Data size: " << jointData.size() << std::endl;
                std::cout << "person: " << ii << "\tjoint: " << jj << std::endl;

                switch (jj){
                case 0:
                    std::cout << "nose" << std::endl;
                    break;
                case 1:
                    std::cout << "left eye" << std::endl;
                    break;
                case 2:
                    std::cout << "right eye" << std::endl;
                    break;
                case 3:
                    std::cout << "left ear" << std::endl;
                    break;
                case 4:
                    std::cout << "right ear" << std::endl;
                    break;
                case 5:
                    std::cout << "left shoulder" << std::endl;
                    break;
                case 6:
                    std::cout << "right shoulder" << std::endl;
                    break;
                case 7:
                    std::cout << "left elbow" << std::endl;
                    break;
                case 8:
                    std::cout << "right elbow" << std::endl;
                    break;
                case 9:
                    std::cout << "left wrist" << std::endl;
                    break;
                case 10:
                    std::cout << "right wrist" << std::endl;
                    break;
                case 11:
                    std::cout << "left hip" << std::endl;
                    break;
                case 12:
                    std::cout << "right hip" << std::endl;
                    break;
                case 13:
                    std::cout << "left knee" << std::endl;
                    break;
                case 14:
                    std::cout << "right knee" << std::endl;
                    break;
                case 15:
                    std::cout << "left ankle" << std::endl;
                    break;
                case 16:
                    std::cout << "right ankle" << std::endl;
                    break;
                default:
                    std::cout << "unknown joint" << std::endl;
                    break;
                }
                for (int kk = 0; kk < jointData.size(); kk++) {
                    float value = jointData[kk];
                    if (kk % 3 == 0) {
                        std::cout << "y coordinate: ";
                        std::cout << value * image_height << std::endl;
                    }
                    else if (kk % 3 == 1) {
                        std::cout << "x coordinate: ";
                        std::cout << value * image_width  << std::endl;
                    }
                    else {
                        std::cout << "confidence: " << value << std::endl;;
                    }                   
                }
            }
        }

        if (false) {
            std::cout << "tensors size: " << interpreter->tensors_size();
            std::cout << "nodes size: " << interpreter->nodes_size();
            std::cout << "inputs: " << interpreter->inputs().size();
            std::cout << "input(0) name: " << interpreter->GetInputName(0);

            int t_size = interpreter->tensors_size();
            for (int i = 0; i < t_size; i++) {
                if (interpreter->tensor(i)->name)
                    std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                    << interpreter->tensor(i)->bytes << ", "
                    << interpreter->tensor(i)->type << ", "
                    << interpreter->tensor(i)->params.scale << ", "
                    << interpreter->tensor(i)->params.zero_point << std::endl;
            }
        }
        // Destory the interpreter earlier than delegates objects.
        interpreter.reset();
 }
}