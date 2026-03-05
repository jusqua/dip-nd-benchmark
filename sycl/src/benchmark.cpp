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

    void operator()(sycl::id<> i) const
    {
        m_output->data[i] = 255 - m_input->data[i];
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

    void operator()(sycl::id<> i) const
    {
        m_output->data[i] = m_input->data[i] > m_threshold ? m_max_value : 0;
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

    void operator()(sycl::id<> i) const
    {
        uint8_t pmin = 255;

        map(i, [&](auto image_index, auto _) {
            pmin = sycl::min(pmin, m_input->data[image_index]);
        });

        m_output->data[i] = pmin;
    }
};

class ConvolveKernel : public WindowKernel {
public:
    using WindowKernel::WindowKernel;

    void operator()(sycl::id<> i) const
    {
        float result = 0.0f;

        map(i, [&](auto image_index, auto window_index) {
            result += m_input->data[image_index] * m_window->data[window_index];
        });

        m_output->data[i] = result;
    }
};

DeviceImage* image_similar_device_from_host(Image* image, sycl::queue& q)
{
    auto d_image = new DeviceImage();
    auto tmp_image = Image();
    d_image->self = sycl::malloc_device<Image>(1, q);

    d_image->data = sycl::malloc_device<uint8_t>(image->size, q);
    tmp_image.data = d_image->data;

    d_image->shape = sycl::malloc_device<int>(image->dimensions + 1, q);
    q.copy(image->shape, d_image->shape, image->dimensions + 1).wait();
    tmp_image.shape = d_image->shape;

    d_image->offset = sycl::malloc_device<int>(image->dimensions + 1, q);
    q.copy(image->offset, d_image->offset, image->dimensions + 1).wait();
    tmp_image.offset = d_image->offset;

    d_image->dimensions = image->dimensions;
    tmp_image.dimensions = d_image->dimensions;
    d_image->size = image->size;
    tmp_image.size = d_image->size;

    q.copy(&tmp_image, d_image->self, 1).wait();

    return d_image;
}

DeviceImage* image_device_from_host(Image* image, sycl::queue& q)
{
    auto d_image = image_similar_device_from_host(image, q);

    q.copy(image->data, d_image->data, image->size).wait();

    return d_image;
}

DeviceImage* image_device_convert_from_host(Image* image, sycl::queue& q)
{
    auto d_image = image_device_from_host(image, q);

    image_destroy(image);

    return d_image;
}

void image_destroy_device(DeviceImage* d_image, sycl::queue& q)
{
    sycl::free(d_image->data, q);
    sycl::free(d_image->shape, q);
    sycl::free(d_image->offset, q);
    sycl::free(d_image->self, q);
    delete d_image;
}

DeviceWindow* window_similar_device_from_host(Window* window, sycl::queue& q)
{
    auto d_window = new DeviceWindow();
    auto tmp_window = Window();
    d_window->self = sycl::malloc_device<Window>(1, q);

    d_window->data = sycl::malloc_device<float>(window->size, q);
    tmp_window.data = d_window->data;

    d_window->shape = sycl::malloc_device<int>(window->dimensions + 1, q);
    q.copy(window->shape, d_window->shape, window->dimensions + 1).wait();
    tmp_window.shape = d_window->shape;

    d_window->offset = sycl::malloc_device<int>(window->dimensions + 1, q);
    q.copy(window->offset, d_window->offset, window->dimensions + 1).wait();
    tmp_window.offset = d_window->offset;

    d_window->dimensions = window->dimensions;
    tmp_window.dimensions = d_window->dimensions;
    d_window->size = window->size;
    tmp_window.size = d_window->size;

    q.copy(&tmp_window, d_window->self, 1).wait();

    return d_window;
}

DeviceWindow* window_device_from_host(Window* window, sycl::queue& q)
{
    auto d_window = window_similar_device_from_host(window, q);

    q.copy(window->data, d_window->data, window->size).wait();

    return d_window;
}

DeviceWindow* window_device_convert_from_host(Window* window, sycl::queue& q)
{
    auto d_window = window_device_from_host(window, q);

    window_destroy(window);

    return d_window;
}

void window_destroy_device(DeviceWindow* d_window, sycl::queue& q)
{
    sycl::free(d_window->data, q);
    sycl::free(d_window->shape, q);
    sycl::free(d_window->offset, q);
    sycl::free(d_window->self, q);
    delete d_window;
}

void benchmark(VglImage* vglimage, size_t rounds, std::function<void(VglImage*, std::string)> save_image)
{
    sycl::queue q;

    auto image = image_from_vglimage(vglimage);
    auto dimensions = image->dimensions;

    auto d_input = image_device_from_host(image, q);
    auto d_output = image_similar_device_from_host(image, q);
    auto d_temp = image_similar_device_from_host(image, q);

    auto d_cross_window = window_device_convert_from_host(window_create_from_type(WindowType::CROSS, dimensions), q);
    auto d_cube_window = window_device_convert_from_host(window_create_from_type(WindowType::CUBE, dimensions), q);
    auto d_mean_window = window_device_convert_from_host(window_create_from_type(WindowType::MEAN, dimensions), q);

    auto d_cube_window_array = new DeviceWindow*[dimensions + 1];
    auto d_mean_window_array = new DeviceWindow*[dimensions + 1];
    {
        auto cube_window_1d = window_create_from_type(WindowType::CUBE, 1);
        auto mean_window_1d = window_create_from_type(WindowType::MEAN, 1);

        for (int i = 1; i <= dimensions; ++i) {
            cube_window_1d->shape[i] = 3;
            mean_window_1d->shape[i] = 3;

            d_cube_window_array[i] = window_device_from_host(cube_window_1d, q);
            d_mean_window_array[i] = window_device_from_host(mean_window_1d, q);

            cube_window_1d->shape[i] = 1;
            mean_window_1d->shape[i] = 1;
        }

        window_destroy(cube_window_1d);
        window_destroy(mean_window_1d);
    }

    auto save_sample = [&](std::string name) {
        q.memcpy(vglimage->getImageData(), d_output->data, d_output->size);
        save_image(vglimage, name);
    };

    auto builder = BenchmarkBuilder();
    builder.attach({
        .name = "upload",
        .type = "group",
        .group = "memory",
        .func = [&] { q.copy(image->data, d_input->data, image->size).wait(); },
    });
    builder.attach({
        .name = "download",
        .type = "group",
        .group = "memory",
        .func = [&] { q.copy(d_input->data, image->data, image->size).wait(); },
    });
    builder.attach({
        .name = "copy",
        .type = "group",
        .group = "memory",
        .post = save_sample,
        .func = [&] { q.copy(d_input->data, d_output->data, image->size).wait(); },
    });
    builder.attach({
        .name = "invert",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { q.parallel_for(image->size, InvertKernel(d_input->self, d_output->self)).wait(); },
    });
    builder.attach({
        .name = "threshold",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { q.parallel_for(image->size, ThresholdKernel(d_input->self, d_output->self, 128, 255)).wait(); },
    });
    builder.attach({
        .name = "erode-cross",
        .type = "single",
        .post = save_sample,
        .func = [&] { q.parallel_for(image->size, ErodeKernel(d_input->self, d_output->self, d_cross_window->self)).wait(); },
    });
    builder.attach({
        .name = "erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] { q.parallel_for(image->size, ErodeKernel(d_input->self, d_output->self, d_cube_window->self)).wait(); },
    });
    builder.attach({
        .name = "split-erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            q.parallel_for(image->size, ErodeKernel(d_input->self, d_temp->self, d_cube_window_array[1]->self)).wait();
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    q.parallel_for(image->size, ErodeKernel(d_output->self, d_temp->self, d_cube_window_array[i]->self)).wait();
                else
                    q.parallel_for(image->size, ErodeKernel(d_temp->self, d_output->self, d_cube_window_array[i]->self)).wait();
            if (dimensions & 0b1)
                q.copy(d_output->data, d_temp->data, image->size).wait();
        },
    });
    builder.attach({
        .name = "convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] { q.parallel_for(image->size, ConvolveKernel(d_input->self, d_output->self, d_mean_window->self)).wait(); },
    });
    builder.attach({
        .name = "split-convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            q.parallel_for(image->size, ConvolveKernel(d_input->self, d_temp->self, d_mean_window_array[1]->self)).wait();
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    q.parallel_for(image->size, ConvolveKernel(d_output->self, d_temp->self, d_mean_window_array[i]->self))
                        .wait();
                else
                    q.parallel_for(image->size, ConvolveKernel(d_temp->self, d_output->self, d_mean_window_array[i]->self))
                        .wait();
            if (dimensions & 0b1)
                q.copy(d_output->data, d_temp->data, image->size).wait();
        },
    });
    builder.run(rounds);

    image_destroy(image);
    image_destroy_device(d_input, q);
    image_destroy_device(d_output, q);
    image_destroy_device(d_temp, q);
    window_destroy_device(d_cross_window, q);
    window_destroy_device(d_cube_window, q);
    window_destroy_device(d_mean_window, q);
    for (auto i = 1; i <= dimensions; ++i) {
        window_destroy_device(d_cube_window_array[i], q);
        window_destroy_device(d_mean_window_array[i], q);
    }
    delete[] d_cube_window_array;
    delete[] d_mean_window_array;
}
