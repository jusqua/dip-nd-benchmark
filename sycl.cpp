#include <cstdio>
#include <print>
#include <sycl/sycl.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>

#include "main.hpp"
#include "utils.hpp"

template <int Dimensions>
class Kernel {
protected:
  uint8_t *input;
  uint8_t *output;
  int *shape;

public:
  Kernel(uint8_t *input, uint8_t *output, int *shape)
    : input(input), output(output), shape(shape) {}

  virtual void operator()(sycl::nd_item<Dimensions> item) const;
};

template <int Dimensions>
class InvertKernel : public Kernel<Dimensions> {
public:
  using Kernel<Dimensions>::Kernel;

  void operator()(sycl::nd_item<Dimensions> item) const {
    size_t i = item.get_global_linear_id();
    this->output[i] = 255 - this->input[i];
  }
};

template <int Dimensions>
class ThresholdKernel : public Kernel<Dimensions> {
private:
  uint8_t threshold;
  uint8_t max_value;

public:
  ThresholdKernel(uint8_t *input, uint8_t *output, int *shape, uint8_t threshold, uint8_t max_value)
    : Kernel<Dimensions>(input, output, shape), threshold(threshold),
      max_value(max_value) {}

  void operator()(sycl::nd_item<Dimensions> item) const {
    size_t i = item.get_global_linear_id();
    this->output[index] =
        this->input[index] > this->threshold ? this->max_value : 0;
  }
};


int computing_units_selector_v(const sycl::device &dev) {
  if (dev.has(sycl::aspect::cpu)) {
    return -1;
  }

  return dev.get_info<sycl::info::device::max_compute_units>();
}

template<int Dimensions>
sycl::nd_range<Dimensions> get_range(int* shape) {
  constexpr size_t wg_size = 16;
  if constexpr (Dimensions == 1) {
		sycl::range global_range(shape[1]);
		sycl::range local_range(wg_size);
		while (global_range[0] % local_range[0] != 0 && local_range[0] > 1) {}
	    local_range[0] /= 2;
		return sycl::nd_range(global_range, local_range);
  } else if constexpr (Dimensions == 2) {
		sycl::range global_range(shape[1], shape[2]);
		sycl::range local_range(wg_size, wg_size);
		for (auto i = 0; i < Dimensions; ++i) {
			while (global_range[i] % local_range[i] != 0 && local_range[i] > 1)
		    local_range[i] /= 2;
		}
		return sycl::nd_range(global_range, local_range);
  } else if constexpr (Dimensions == 3) {
		sycl::range global_range(shape[1], shape[2], shape[3]);
		sycl::range local_range(wg_size, wg_size, wg_size);
		for (auto i = 0; i < Dimensions; ++i) {
			while (global_range[i] % local_range[i] != 0 && local_range[i] > 1)
		    local_range[i] /= 2;
		}
		return sycl::nd_range(global_range, local_range);
  } else if constexpr (Dimensions == 4) {
		sycl::range global_range(shape[1], shape[2], shape[3], shape[4]);
		sycl::range local_range(wg_size, wg_size, wg_size, wg_size);
		for (auto i = 0; i < Dimensions; ++i) {
			while (global_range[i] % local_range[i] != 0 && local_range[i] > 1)
		    local_range[i] /= 2;
		}
		return sycl::nd_range(global_range, local_range);
  } else if constexpr (Dimensions == 5) {
		sycl::range global_range(shape[1], shape[2], shape[3], shape[4], shape[5]);
		sycl::range local_range(wg_size, wg_size, wg_size, wg_size, wg_size);
		for (auto i = 0; i < Dimensions; ++i) {
			while (global_range[i] % local_range[i] != 0 && local_range[i] > 1)
		    local_range[i] /= 2;
		}
		return sycl::nd_range(global_range, local_range);
  } else {
  	std::println(stderr, "{}D not supported", Dimensions);
   	std::exit(1);
  }
}

template<int Dimensions>
void nd_benchmark(VglImage *input, size_t rounds, std::function<void(VglImage *, const char *)> save_image) {
  VglImage *output = vglCreateImage(input);

  auto q = sycl::queue{computing_units_selector_v};
  if (!q.get_device().has(sycl::aspect::gpu)) {
    std::println(stderr, "Error: No GPU device found, aborting");
    return;
  }

  auto is_usm_compatible = q.get_device().has(sycl::aspect::usm_device_allocations);
  if (!is_usm_compatible) {
    std::println(stderr, "Error: Device does not support USM device allocations, aborting");
    return;
  }

  std::println("Device: {}", q.get_device().get_info<sycl::info::device::name>());
  std::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

  auto total_size = input->getTotalSizeInBytes();
  uint8_t *d_input = sycl::malloc_device<uint8_t>(total_size, q);
  uint8_t *d_output = sycl::malloc_device<uint8_t>(total_size, q);

  q.memcpy(d_input, input->getImageData(), total_size).wait();

  auto save_sample = [&](std::string name) {
	  q.memcpy(output->getImageData(), d_input, total_size).wait();
	 	save_image(output, name.c_str());
  };

  auto kernel_range = get_range<Dimensions>(input->shape);

  auto builder = BenchmarkBuilder();
  builder.attach({
    .name = "upload",
    .func = [&] {
      q.memcpy(d_input, input->getImageData(), total_size).wait();
    }
  });
  builder.attach({
	  .name = "download",
		.func = [&] {
      q.memcpy(output->getImageData(), d_input, total_size).wait();
    }
  });
  builder.attach({
	  .name = "copy",
		.func = [&] {
      q.memcpy(d_output, d_input, total_size).wait();
    },
    .post = save_sample
  });

  builder.run(rounds);

  sycl::free(d_input, q);
  sycl::free(d_output, q);
  free(output);
}

void benchmark(VglImage *image, size_t rounds, std::function<void(VglImage *, const char *)> save_image) {
	switch (image->ndim) {
		case 1:
			nd_benchmark<1>(image, rounds, save_image);
			break;
		case 2:
			nd_benchmark<2>(image, rounds, save_image);
			break;
		case 3:
			nd_benchmark<3>(image, rounds, save_image);
			break;
		case 4:
			nd_benchmark<4>(image, rounds, save_image);
			break;
		case 5:
			nd_benchmark<5>(image, rounds, save_image);
			break;
		default:
			std::println(stderr, "{}D not supported", image->ndim);
	   	std::exit(1);
			return;
			break;
	}
}
