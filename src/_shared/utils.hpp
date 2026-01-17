#pragma once

#ifndef DIP_ND_BENCHMARK_UTILS_HPP
#define DIP_ND_BENCHMARK_UTILS_HPP

#include <chrono>
#include <functional>
#include <print>
#include <string>
#include <vector>
#include <functional>

#include <visiongl/image.hpp>

struct Image {
	uint8_t *data;
	int *shape;
  int *offset;
  uint8_t dimensions;
  size_t size;

  Image() : data(nullptr), shape(nullptr), offset(nullptr), dimensions(0), size(0) {}

  Image(uint8_t *data, int *shape, int *offset, uint8_t dimensions, size_t size)
  	: data(data), shape(shape), offset(offset), dimensions(dimensions), size(size) {}
};

struct Window {
	float *data;
	int *shape;
  int *offset;
  uint8_t dimensions;
  size_t size;

  Window() : data(nullptr), shape(nullptr), offset(nullptr), dimensions(0), size(0) {}

  Window(float *data, int *shape, int *offset, uint8_t dimensions, size_t size)
  	: data(data), shape(shape), offset(offset), dimensions(dimensions), size(size) {}
};

struct BenchmarkSpec {
  std::string name;
  std::function<void(void)> func;
  std::function<void(std::string)> post;
};

class BenchmarkBuilder {
private:
  std::vector<BenchmarkSpec> specs;

  void perform_benchmark(std::size_t rounds, const BenchmarkSpec &spec) {
    auto time_start_once = std::chrono::high_resolution_clock::now();
    spec.func();
    auto time_end_once = std::chrono::high_resolution_clock::now();
    double once_duration =
        std::chrono::duration<double>(time_end_once - time_start_once).count();

    std::print("{},{}", spec.name, once_duration);

    if (rounds <= 1) {
      std::println();
      return;
    }

    auto mean_start_times = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < rounds; ++i)
      spec.func();
    auto mean_end_times = std::chrono::high_resolution_clock::now();

    double mean_duration =
        std::chrono::duration<double>(mean_end_times - mean_start_times)
            .count() /
        rounds;
    std::println(",{}", mean_duration);
  }

public:
  inline void attach(BenchmarkSpec &&spec) { specs.emplace_back(spec); }

  inline void run(std::size_t rounds) {
    std::println("operator,once{}", rounds <= 1 ? "" : ",mean");
    for (const auto &spec : specs) {
      perform_benchmark(rounds, spec);
      if (spec.post != nullptr)
        spec.post(spec.name);
    }
  }
};


void benchmark(VglImage *image, size_t rounds, std::function<void(VglImage *, const char *)> save_image);

#endif // DIP_ND_BENCHMARK_UTILS_HPP
