#pragma once
#ifndef TFLITE_MOVENET_TRACKING_H_
#define TFLITE_MOVENET_TRACKING_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

namespace movenet_tracking {

	struct Settings {
		bool verbose = false;
		bool accel = false;
		TfLiteType input_type = kTfLiteFloat32;
		bool profiling = false;
		bool allow_fp16 = false;
		bool gl_backend = false;
		bool hexagon_delegate = false;
		bool xnnpack_delegate = false;
		int loop_count = 1;
		float input_mean = 127.5f;
		float input_std = 127.5f;
		tflite::string model_name = "./model.tflite";
		tflite::FlatBufferModel* model;
		tflite::string input_jpg_name = "./input_image.bmp";
		tflite::string labels_file_name = "./labels.txt";
		int number_of_threads = 4;
		int number_of_results = 5;
		int max_profiling_buffer_entries = 1024;
		int number_of_warmup_runs = 2;
	};

	void RunInference(Settings* settings);
}  // namespace movenet_tracking
#endif