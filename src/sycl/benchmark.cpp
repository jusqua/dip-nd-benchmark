#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <print>

#include <sycl/sycl.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>
#include <visiongl/strel.hpp>

#include "../_shared/utils.hpp"

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
	Window *window;

	template<typename Func>
	inline void map(const size_t index, Func&& func) const {
		int image_coord[VGL_ARR_SHAPE_SIZE];
		int window_coord[VGL_ARR_SHAPE_SIZE];
		int ires = index;
		int idim = 0;
		uint8_t pmin = 255;

		for(int d = this->input->dimensions; d >= 1; --d) {
	    int off = this->input->offset[d];
	    idim = ires / off;
	    ires = ires - idim * off;
	    image_coord[d] = idim - (this->window->shape[d] - 1) / 2;
	  }

	  int image_index = 0;
	  for(int window_index = 0; window_index < this->window->size; ++window_index) {
	    if (this->window->data[window_index] == 0) continue;

	    ires = window_index;
	    image_index = 0;

	    for(int d = this->input->dimensions; d > this->window->dimensions; --d)
	      image_index += this->input->offset[d] * image_coord[d];

	    for(int d = this->window->dimensions; d >= 1; --d) {
	      int off = window->offset[d];
	      idim = ires / off;
	      ires = ires - idim * off;
	      window_coord[d] = idim + image_coord[d];
	      window_coord[d] = sycl::clamp(window_coord[d], 0, this->input->shape[d] - 1);

	      image_index += this->input->offset[d] * window_coord[d];
	    }

	    func(image_index, window_index);
	  }
	}

public:
  WindowKernel(Image *input, Image *output, Window *window)
    : Kernel(input, output), window(window) {}
};

class ErodeKernel : public WindowKernel {
public:
	using WindowKernel::WindowKernel;

  void operator()(sycl::id<> item) const {
		const size_t i = item.get(0);

		uint8_t pmin = std::numeric_limits<uint8_t>::max();
		map(i, [&](size_t image_index, size_t _) {
	    pmin = sycl::min(pmin, this->input->data[image_index]);
	  });

	  this->output->data[i] = pmin;
  }
};

class ConvolveKernel : public WindowKernel {
public:
	using WindowKernel::WindowKernel;

  void operator()(sycl::id<> item) const {
		const size_t i = item.get(0);

		float result = 0.0f;
		map(i, [&](size_t image_index, size_t window_index) {
			result += this->input->data[image_index] * this->window->data[window_index];
	  });

	  this->output->data[i] = result;
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
  auto d_aux_data = sycl::malloc_device<uint8_t>(image_size, q);

  q.memcpy(d_shape, image->vglShape->shape, dimensions_size).wait();
  q.memcpy(d_offset, image->vglShape->offset, dimensions_size).wait();
  q.memcpy(d_input_data, image->getImageData(), image_size).wait();

  auto input = Image(d_input_data, d_shape, d_offset, dimensions, image_size);
  auto output = Image(d_output_data, d_shape, d_offset, dimensions, image_size);
  auto aux = Image(d_aux_data, d_shape, d_offset, dimensions, image_size);

  auto strel_cross = new VglStrEl(VGL_STREL_CROSS, dimensions);
  auto strel_cube = new VglStrEl(VGL_STREL_CUBE, dimensions);
  auto strel_mean = new VglStrEl(VGL_STREL_MEAN, dimensions);
  auto window_size = strel_cross->vglShape->size * sizeof(float);

  auto d_window_cross_data = sycl::malloc_device<float>(window_size, q);
  auto d_window_cube_data = sycl::malloc_device<float>(window_size, q);
  auto d_window_mean_data = sycl::malloc_device<float>(window_size, q);
  auto d_window_shape = sycl::malloc_device<int>(dimensions_size, q);
  auto d_window_offset = sycl::malloc_device<int>(dimensions_size, q);

  q.memcpy(d_window_cross_data, strel_cross->data, window_size).wait();
  q.memcpy(d_window_cube_data, strel_cube->data, window_size).wait();
  q.memcpy(d_window_mean_data, strel_mean->data, window_size).wait();
  q.memcpy(d_window_shape, strel_mean->vglShape->shape, dimensions_size).wait();
  q.memcpy(d_window_offset, strel_mean->vglShape->offset, dimensions_size).wait();

  auto window_cross = Window(d_window_cross_data, d_window_shape, d_window_offset, dimensions, window_size);
  auto window_cube = Window(d_window_cube_data, d_window_shape, d_window_offset, dimensions, window_size);
  auto window_mean = Window(d_window_mean_data, d_window_shape, d_window_offset, dimensions, window_size);

  std::array<Window, VGL_ARR_SHAPE_SIZE> window_cube_array;
  std::array<Window, VGL_ARR_SHAPE_SIZE> window_mean_array;
  auto window_linear_size = 3 * sizeof(float);
  float window_cube_data_linear[] = {1.0f, 1.0f, 1.0f};
  float window_mean_data_linear[] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
  auto d_window_cube_linear_data = sycl::malloc_device<float>(window_linear_size, q);
  auto d_window_mean_linear_data = sycl::malloc_device<float>(window_linear_size, q);
  q.memcpy(d_window_cube_linear_data, window_cube_data_linear, window_linear_size).wait();
  q.memcpy(d_window_mean_linear_data, window_mean_data_linear, window_linear_size).wait();
  int window_linear_shape[VGL_ARR_SHAPE_SIZE];
  for (int i = 0; i < VGL_ARR_SHAPE_SIZE; ++i) window_linear_shape[i] = 1;

  for (int i = 1; i <= dimensions; ++i) {
	  window_linear_shape[i] = 3;
		auto vgl_shape = new VglShape(window_linear_shape, dimensions);

	  auto d_window_linear_shape = sycl::malloc_device<int>(dimensions_size, q);
	  auto d_window_linear_offset = sycl::malloc_device<int>(dimensions_size, q);
    q.memcpy(d_window_linear_shape, vgl_shape->shape, dimensions_size).wait();
    q.memcpy(d_window_linear_offset, vgl_shape->offset, dimensions_size).wait();

    window_cube_array[i] = Window(d_window_cube_linear_data, d_window_linear_shape, d_window_linear_offset, dimensions, window_linear_size);
    window_mean_array[i] = Window(d_window_mean_linear_data, d_window_linear_shape, d_window_linear_offset, dimensions, window_linear_size);

    delete vgl_shape;
    window_linear_shape[i] = 1;
  }

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
  builder.attach({
    .name = "split-erode-cube",
    .func = [&] {
      q.parallel_for(image_size, ErodeKernel(&input, &aux, &window_cube_array[1])).wait();
      for (int i = 2; i <= dimensions; ++i)
        if (i & 0b1) q.parallel_for(image_size, ErodeKernel(&output, &aux, &window_cube_array[i])).wait();
        else q.parallel_for(image_size, ErodeKernel(&aux, &output, &window_cube_array[i])).wait();
      if (dimensions & 0b1) q.memcpy(aux.data, output.data, image_size).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "convolve",
    .func = [&] {
      q.parallel_for(image_size, ConvolveKernel(&input, &output, &window_mean)).wait();
    },
    .post = save_sample
  });
  builder.attach({
    .name = "split-convolve",
    .func = [&] {
      q.parallel_for(image_size, ConvolveKernel(&input, &aux, &window_mean_array[1])).wait();
      for (int i = 2; i <= dimensions; ++i)
        if (i & 0b1) q.parallel_for(image_size, ConvolveKernel(&output, &aux, &window_mean_array[i])).wait();
        else q.parallel_for(image_size, ConvolveKernel(&aux, &output, &window_mean_array[i])).wait();
      if (dimensions & 0b1) q.memcpy(aux.data, output.data, image_size).wait();
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
  sycl::free(d_window_cube_linear_data, q);
  sycl::free(d_window_mean_linear_data, q);
  for (auto i = 1; i <= dimensions; ++i) {
	  sycl::free(window_cube_array[i].shape, q);
	  sycl::free(window_cube_array[i].offset, q);
  }
  delete strel_cross;
  delete strel_cube;
  delete strel_mean;
  delete tmp;
}
