#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <print>

#include <sycl/sycl.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>
#include <visiongl/strel.hpp>

#include "main.hpp"
#include "utils.hpp"

struct Image {
	uint8_t *data;
	int *shape;
  int *offset;
  uint8_t dimensions;
  size_t size;

  Image(uint8_t *data, int *shape, int *offset, uint8_t dimensions, size_t size)
  	: data(data), shape(shape), offset(offset), dimensions(dimensions), size(size) {}
};

class Kernel {
protected:
	Image *input;
	Image *output;

public:
  Kernel(Image *input, Image *output) : input(input), output(output) {}

  virtual void operator()(sycl::id<> item) const {}
};

class InvertKernel : public Kernel {
public:
  using Kernel::Kernel;

  void operator()(sycl::id<> item) const {
    const size_t i = item.get(0);
    this->output->data[i] = std::numeric_limits<uint8_t>::max() - this->input->data[i];
  }
};

class ThresholdKernel : public Kernel {
private:
  uint8_t threshold;
  uint8_t max_value;

public:
  ThresholdKernel(Image *input, Image *output, uint8_t threshold = std::numeric_limits<uint8_t>::max() / 2, uint8_t max_value = std::numeric_limits<uint8_t>::max())
    : Kernel(input, output), threshold(threshold), max_value(max_value) {}

  void operator()(sycl::id<> item) const {
	  const size_t i = item.get(0);
    this->output->data[i] = this->input->data[i] > this->threshold ? this->max_value : 0;
  }
};

class WindowKernel : public Kernel {
protected:
	Image *window;

public:
  WindowKernel(Image *input, Image *output, Image *window)
    : Kernel(input, output), window(window) {}
};

class ErodeKernel : public WindowKernel {
public:
	using WindowKernel::WindowKernel;

  void operator()(sycl::id<> item) const {
		const size_t i = item.get(0);

		int image_coord[VGL_ARR_SHAPE_SIZE];
		int window_coord[VGL_ARR_SHAPE_SIZE];
		int ires = i;
		int idim = 0;
		uint8_t pmin = 255;

		for(int d = this->input->dimensions; d >= 1; --d) {
	    int off = this->input->offset[d];
	    idim = ires / off;
	    ires = ires - idim * off;
	    image_coord[d] = idim - (this->window->shape[d] - 1) / 2;
	  }

	  int pos = 0;
	  for(int i = 0; i < this->window->size; ++i) {
	    if (this->window->data[i] == 0) continue;

	    ires = i;
	    pos = 0;

	    for(int d = this->input->dimensions; d > this->window->dimensions; --d)
	      pos += this->input->offset[d] * image_coord[d];

	    for(int d = this->window->dimensions; d >= 1; --d) {
	      int off = window->offset[d];
	      idim = ires / off;
	      ires = ires - idim * off;
	      window_coord[d] = idim + image_coord[d];
	      window_coord[d] = sycl::clamp(window_coord[d], 0, this->input->shape[d] - 1);

	      pos += this->input->offset[d] * window_coord[d];
	    }

	    pmin = sycl::min(pmin, this->input->data[pos]);
	  }

	  this->output->data[i] = pmin;
  }
};

void benchmark(VglImage *image, size_t rounds, std::function<void(VglImage *, const char *)> save_image) {
	sycl::queue q;

  std::println("Device: {}", q.get_device().get_info<sycl::info::device::name>());
  std::println("Platform: {}", q.get_device().get_platform().get_info<sycl::info::platform::name>());

  auto tmp = vglCreateImage(image);

  auto dimensions = image->ndim;
  auto dimensions_size = sizeof(int) * (dimensions + 1);
  auto d_offset = sycl::malloc_device<int>(dimensions_size, q);
  auto d_shape = sycl::malloc_device<int>(dimensions_size, q);

  auto image_size = image->vglShape->size;
  auto d_input_data = sycl::malloc_device<uint8_t>(image_size, q);
  auto d_output_data = sycl::malloc_device<uint8_t>(image_size, q);

  q.memcpy(d_offset, image->vglShape->shape, dimensions_size).wait();
  q.memcpy(d_shape, image->vglShape->offset, dimensions_size).wait();
  q.memcpy(d_input_data, image->getImageData(), image_size).wait();

  auto input = Image(d_input_data, d_shape, d_offset, dimensions, image_size);
  auto output = Image(d_output_data, d_shape, d_offset, dimensions, image_size);

  auto strel_cross = new VglStrEl(VGL_STREL_CROSS, dimensions);
  auto strel_cube = new VglStrEl(VGL_STREL_CUBE, dimensions);
  auto strel_mean = new VglStrEl(VGL_STREL_MEAN, dimensions);
  auto window_size = strel_cross->vglShape->size;

  auto d_window_cross_data = sycl::malloc_device<uint8_t>(window_size, q);
  auto d_window_cube_data = sycl::malloc_device<uint8_t>(window_size, q);
  auto d_window_mean_data = sycl::malloc_device<uint8_t>(window_size, q);
  auto d_window_shape = sycl::malloc_device<int>(dimensions_size, q);
  auto d_window_offset = sycl::malloc_device<int>(dimensions_size, q);

  q.memcpy(d_window_cross_data, strel_cross->data, window_size).wait();
  q.memcpy(d_window_cube_data, strel_cube->data, window_size).wait();
  q.memcpy(d_window_mean_data, strel_mean->data, window_size).wait();
  q.memcpy(d_window_shape, strel_mean->vglShape->shape, dimensions_size).wait();
  q.memcpy(d_window_offset, strel_mean->vglShape->offset, dimensions_size).wait();

  auto window_cross = Image(d_window_cross_data, d_window_shape, d_window_offset, dimensions, window_size);
  auto window_cube = Image(d_window_cube_data, d_window_shape, d_window_offset, dimensions, window_size);
  auto window_mean = Image(d_window_mean_data, d_window_shape, d_window_offset, dimensions, window_size);

  auto save_sample = [&](std::string name) {
	  q.memcpy(tmp->getImageData(), d_output_data, image_size).wait();
	 	save_image(tmp, name.c_str());
  };

  auto builder = BenchmarkBuilder();
  builder.attach({
    .name = "upload",
    .func = [&] {
      q.memcpy(d_input_data, image->getImageData(), image_size).wait();
    }
  });
  builder.attach({
	  .name = "download",
		.func = [&] {
      q.memcpy(tmp->getImageData(), d_input_data, image_size).wait();
    }
  });
  builder.attach({
	  .name = "copy",
		.func = [&] {
      q.memcpy(d_output_data, d_input_data, image_size).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "invert",
    .func = [&] {
      q.parallel_for(image_size, InvertKernel(&input, &output)).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "threshold",
    .func = [&] {
      q.parallel_for(image_size, ThresholdKernel(&input, &output)).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "erode-cross",
    .func = [&] {
      q.parallel_for(image_size, ErodeKernel(&input, &output, &window_cross)).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "erode-cube",
    .func = [&] {
      q.parallel_for(image_size, ErodeKernel(&input, &output, &window_cube)).wait();
    },
    .post = save_sample
  });

  builder.run(rounds);

  sycl::free(d_window_cross_data, q);
  sycl::free(d_window_cube_data, q);
  sycl::free(d_window_mean_data, q);
  sycl::free(d_window_shape, q);
  sycl::free(d_window_offset, q);
  sycl::free(d_offset, q);
  sycl::free(d_shape, q);
  sycl::free(d_input_data, q);
  sycl::free(d_output_data, q);
  delete strel_cross;
  delete strel_cube;
  delete strel_mean;
  delete tmp;
}
