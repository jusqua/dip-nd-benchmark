#include <cstdio>
#include <limits>
#include <print>
#include <sycl/sycl.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>

#include "main.hpp"
#include "utils.hpp"

class Kernel {
protected:
  uint8_t *input;
  uint8_t *output;
  int *shape;
  int dimensions;

public:
  Kernel(uint8_t *input, uint8_t *output, int *shape, int dimensions)
    : input(input), output(output), shape(shape), dimensions(dimensions) {}

  virtual void operator()(sycl::id<> item) const {}
};

class InvertionKernel : public Kernel {
public:
  using Kernel::Kernel;

  void operator()(sycl::id<> item) const {
    const size_t i = item.get(0);
    this->output[i] = std::numeric_limits<uint8_t>::max() - this->input[i];
  }
};

class ThresholdKernel : public Kernel {
private:
  uint8_t threshold;
  uint8_t max_value;

public:
  ThresholdKernel(uint8_t *input, uint8_t *output, int *shape, int dimensions, uint8_t threshold = std::numeric_limits<uint8_t>::max() / 2, uint8_t max_value = std::numeric_limits<uint8_t>::max())
    : Kernel(input, output, shape, dimensions), threshold(threshold), max_value(max_value) {}

  void operator()(sycl::id<> item) const {
	  const size_t i = item.get(0);
    this->output[i] = this->input[i] > this->threshold ? this->max_value : 0;
  }
};

void benchmark(VglImage *input, size_t rounds, std::function<void(VglImage *, const char *)> save_image) {
	sycl::queue q;

  std::println("Device: {}", q.get_device().get_info<sycl::info::device::name>());
  std::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

  auto output = vglCreateImage(input);
  auto dimensions = input->ndim;
  auto dimensions_size = sizeof(int) * (dimensions + 1);
  auto d_shape = sycl::malloc_device<int>(dimensions_size, q);
  auto image_size = input->getTotalSizeInBytes();
  auto d_input = sycl::malloc_device<uint8_t>(image_size, q);
  auto d_output = sycl::malloc_device<uint8_t>(image_size, q);

  q.memcpy(d_shape, input->shape, dimensions_size).wait();
  q.memcpy(d_input, input->getImageData(), image_size).wait();

  auto save_sample = [&](std::string name) {
	  q.memcpy(output->getImageData(), d_output, image_size).wait();
	 	save_image(output, name.c_str());
  };

  auto builder = BenchmarkBuilder();
  builder.attach({
    .name = "upload",
    .func = [&] {
      q.memcpy(d_input, input->getImageData(), image_size).wait();
    }
  });
  builder.attach({
	  .name = "download",
		.func = [&] {
      q.memcpy(output->getImageData(), d_input, image_size).wait();
    }
  });
  builder.attach({
	  .name = "copy",
		.func = [&] {
      q.memcpy(d_output, d_input, image_size).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "invertion",
    .func = [&] {
      q.parallel_for(image_size, InvertionKernel(d_input, d_output, d_shape, dimensions)).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "threshold",
    .func = [&] {
      q.parallel_for(image_size, ThresholdKernel(d_input, d_output, d_shape, dimensions)).wait();
    },
    .post = save_sample
  });

  builder.run(rounds);

  sycl::free(d_shape, q);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  free(output);
}
