#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <sycl/sycl.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>
#include <visiongl/strel.hpp>

#include <utils.hpp>

class Kernel {
protected:
    Image* m_input;
    Image* m_output;

public:
    Kernel(Image* input, Image* output)
        : m_input(input)
        , m_output(output)
    {
    }
};

class InvertKernel : public Kernel {
public:
    using Kernel::Kernel;

    void operator()(sycl::id<> item) const
    {
        size_t const index = item.get(0);
        m_output->data[index] = 255 - m_input->data[index];
    }
};

class ThresholdKernel : public Kernel {
private:
    uint8_t m_threshold;
    uint8_t m_max_value;

public:
    ThresholdKernel(Image* input, Image* output, uint8_t threshold, uint8_t max_value)
        : Kernel(input, output)
        , m_threshold(threshold)
        , m_max_value(max_value)
    {
    }

    void operator()(sycl::id<> item) const
    {
        size_t const index = item.get(0);
        m_output->data[index] = m_input->data[index] > m_threshold ? m_max_value : 0;
    }
};

class WindowKernel : public Kernel {
protected:
    Window* m_window;

public:
    WindowKernel(Image* input, Image* output, Window* window)
        : Kernel(input, output)
        , m_window(window)
    {
    }

    template<typename Func = std::function<void(size_t, size_t)>>
    inline auto map(size_t index, Func&& apply) const
    {
        int image_coord[VGL_ARR_SHAPE_SIZE];
        int window_coord[VGL_ARR_SHAPE_SIZE];
        int ires = index;
        int idim = 0;

        for (int d = m_input->dimensions; d >= 1; --d) {
            int off = m_input->offset[d];
            idim = ires / off;
            ires = ires - idim * off;
            image_coord[d] = idim - (m_window->shape[d] - 1) / 2;
        }

        size_t image_index = 0;
        for (size_t window_index = 0; window_index < m_window->size; ++window_index) {
            if (m_window->data[window_index] == 0)
                continue;

            ires = window_index;
            image_index = 0;

            for (int d = m_input->dimensions; d > m_window->dimensions; --d)
                image_index += m_input->offset[d] * image_coord[d];

            for (int d = m_window->dimensions; d >= 1; --d) {
                int off = m_window->offset[d];
                idim = ires / off;
                ires = ires - idim * off;
                window_coord[d] = idim + image_coord[d];
                window_coord[d] = sycl::clamp(window_coord[d], 0, m_input->shape[d] - 1);

                image_index += m_input->offset[d] * window_coord[d];
            }

            apply(image_index, window_index);
        }
    }
};

class ErodeKernel : public WindowKernel {
public:
    using WindowKernel::WindowKernel;

    void operator()(sycl::id<> item) const
    {
        size_t const index = item.get(0);
        uint8_t pmin = 255;

        map(index, [&](auto image_index, auto _) {
            pmin = sycl::min(pmin, m_input->data[image_index]);
        });

        m_output->data[index] = pmin;
    }
};

class ConvolveKernel : public WindowKernel {
public:
    using WindowKernel::WindowKernel;

    void operator()(sycl::id<> item) const
    {
        size_t const index = item.get(0);
        float result = 0.0f;

        map(index, [&](auto image_index, auto window_index) {
            result += m_input->data[image_index] * m_window->data[window_index];
        });

        m_output->data[index] = result;
    }
};

void benchmark(VglImage* image, size_t rounds, std::function<void(VglImage*, char const*)> save_image)
{
    sycl::queue q;

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
    auto d_input = sycl::malloc_device<Image>(sizeof(Image), q);
    auto d_output = sycl::malloc_device<Image>(sizeof(Image), q);
    auto d_aux = sycl::malloc_device<Image>(sizeof(Image), q);
    q.memcpy(d_input, &input, sizeof(Image)).wait();
    q.memcpy(d_output, &output, sizeof(Image)).wait();
    q.memcpy(d_aux, &aux, sizeof(Image)).wait();

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
    auto d_window_cross = sycl::malloc_device<Window>(sizeof(Window), q);
    auto d_window_cube = sycl::malloc_device<Window>(sizeof(Window), q);
    auto d_window_mean = sycl::malloc_device<Window>(sizeof(Window), q);
    q.memcpy(d_window_cross, &window_cross, sizeof(Window)).wait();
    q.memcpy(d_window_cube, &window_cube, sizeof(Window)).wait();
    q.memcpy(d_window_mean, &window_mean, sizeof(Window)).wait();

    Window window_cube_array[VGL_ARR_SHAPE_SIZE];
    Window window_mean_array[VGL_ARR_SHAPE_SIZE];
    auto window_linear_size = 3 * sizeof(float);
    float window_cube_data_linear[] = { 1.0f, 1.0f, 1.0f };
    float window_mean_data_linear[] = { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f };
    Window* window_cube_array_d[VGL_ARR_SHAPE_SIZE];
    Window* window_mean_array_d[VGL_ARR_SHAPE_SIZE];
    auto d_window_cube_linear_data = sycl::malloc_device<float>(window_linear_size, q);
    auto d_window_mean_linear_data = sycl::malloc_device<float>(window_linear_size, q);
    q.memcpy(d_window_cube_linear_data, window_cube_data_linear, window_linear_size).wait();
    q.memcpy(d_window_mean_linear_data, window_mean_data_linear, window_linear_size).wait();
    int window_linear_shape[VGL_ARR_SHAPE_SIZE];
    for (int i = 0; i < VGL_ARR_SHAPE_SIZE; ++i)
        window_linear_shape[i] = 1;

    for (int i = 1; i <= dimensions; ++i) {
        window_linear_shape[i] = 3;
        auto vgl_shape = new VglShape(window_linear_shape, dimensions);

        auto d_window_linear_shape = sycl::malloc_device<int>(dimensions_size, q);
        auto d_window_linear_offset = sycl::malloc_device<int>(dimensions_size, q);
        q.memcpy(d_window_linear_shape, vgl_shape->shape, dimensions_size).wait();
        q.memcpy(d_window_linear_offset, vgl_shape->offset, dimensions_size).wait();

        window_cube_array[i] = Window(d_window_cube_linear_data, d_window_linear_shape, d_window_linear_offset, dimensions, window_linear_size);
        window_cube_array_d[i] = sycl::malloc_device<Window>(sizeof(Window), q);
        q.memcpy(window_cube_array_d[i], &window_cube_array[i], sizeof(Window)).wait();
        window_mean_array[i] = Window(d_window_mean_linear_data, d_window_linear_shape, d_window_linear_offset, dimensions, window_linear_size);
        window_mean_array_d[i] = sycl::malloc_device<Window>(sizeof(Window), q);
        q.memcpy(window_mean_array_d[i], &window_mean_array[i], sizeof(Window)).wait();

        delete vgl_shape;
        window_linear_shape[i] = 1;
    }

    auto save_sample = [&](std::string name) {
        q.memcpy(tmp->getImageData(), d_output_data, image_size).wait();
        save_image(tmp, name.c_str());
    };

    auto builder = BenchmarkBuilder();
    builder.attach({ .name = "upload", .func = [&] { q.memcpy(d_input_data, image->getImageData(), image_size).wait(); } });
    builder.attach({ .name = "download", .func = [&] { q.memcpy(tmp->getImageData(), d_input_data, image_size).wait(); } });
    builder.attach({ .name = "copy", .func = [&] { q.memcpy(d_output_data, d_input_data, image_size).wait(); }, .post = save_sample });
    builder.attach({ .name = "invert", .func = [&] { q.parallel_for(image_size, InvertKernel(d_input, d_output)).wait(); }, .post = save_sample });
    builder.attach({ .name = "threshold", .func = [&] { q.parallel_for(image_size, ThresholdKernel(d_input, d_output, 128, 255)).wait(); }, .post = save_sample });
    builder.attach({ .name = "erode-cross", .func = [&] { q.parallel_for(image_size, ErodeKernel(d_input, d_output, d_window_cross)).wait(); }, .post = save_sample });
    builder.attach({ .name = "erode-cube", .func = [&] { q.parallel_for(image_size, ErodeKernel(d_input, d_output, d_window_cube)).wait(); }, .post = save_sample });
    builder.attach({
        .name = "split-erode-cube",
        .func = [&] {
            q.parallel_for(image_size, ErodeKernel(d_input, d_aux, window_cube_array_d[1])).wait();
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    q.parallel_for(image_size, ErodeKernel(d_output, d_aux, window_cube_array_d[i])).wait();
                else
                    q.parallel_for(image_size, ErodeKernel(d_aux, d_output, window_cube_array_d[i])).wait();
            if (dimensions & 0b1)
                q.memcpy(d_aux_data, d_output_data, image_size).wait();
        },
        .post = save_sample,
    });
    builder.attach({
        .name = "convolve",
        .func = [&] { q.parallel_for(image_size, ConvolveKernel(d_input, d_output, d_window_mean)).wait(); },
        .post = save_sample,
    });
    builder.attach({
        .name = "split-convolve",
        .func = [&] {
            q.parallel_for(image_size, ConvolveKernel(d_input, d_aux, window_mean_array_d[1])).wait();
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    q.parallel_for(image_size, ConvolveKernel(d_output, d_aux, window_mean_array_d[i]))
                        .wait();
                else
                    q.parallel_for(image_size, ConvolveKernel(d_aux, d_output, window_mean_array_d[i]))
                        .wait();
            if (dimensions & 0b1)
                q.memcpy(d_aux_data, d_output_data, image_size).wait();
        },
        .post = save_sample,
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
    sycl::free(d_window_cross, q);
    sycl::free(d_window_cube, q);
    sycl::free(d_window_mean, q);
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_aux, q);
    for (auto i = 1; i <= dimensions; ++i) {
        sycl::free(window_cube_array[i].shape, q);
        sycl::free(window_cube_array[i].offset, q);
        sycl::free(window_cube_array_d[i], q);
        sycl::free(window_mean_array_d[i], q);
    }
    delete strel_cross;
    delete strel_cube;
    delete strel_mean;
    delete tmp;
}
