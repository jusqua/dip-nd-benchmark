#include <cuda_runtime.h>

#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>
#include <visiongl/strel.hpp>

#include <utils.hpp>

__global__ void invert_kernel(Image const* input, Image* output)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;
    output->data[i] = 255 - input->data[i];
}

__global__ void threshold_kernel(Image const* input, Image* output, uint8_t threshold, uint8_t max_value)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;
    output->data[i] = input->data[i] > threshold ? max_value : 0;
}

template<typename Func = std::function<void(size_t, size_t)>>
__device__ void window_map(Image* input, Window* window, size_t const index, Func&& func)
{
    int image_coord[VGL_ARR_SHAPE_SIZE];
    int window_coord[VGL_ARR_SHAPE_SIZE];
    int ires = index;
    int idim = 0;

    for (int d = input->dimensions; d >= 1; --d) {
        int off = input->offset[d];
        idim = ires / off;
        ires = ires - idim * off;
        image_coord[d] = idim - (window->shape[d] - 1) / 2;
    }

    size_t image_index = 0;
    for (size_t window_index = 0; window_index < window->size; ++window_index) {
        if (window->data[window_index] == 0)
            continue;

        ires = (int)window_index;
        image_index = 0;

        for (int d = input->dimensions; d > window->dimensions; --d)
            image_index += input->offset[d] * image_coord[d];

        for (int d = window->dimensions; d >= 1; --d) {
            int off = window->offset[d];
            idim = ires / off;
            ires = ires - idim * off;
            window_coord[d] = idim + image_coord[d];
            int maxv = input->shape[d] - 1;
            if (window_coord[d] < 0)
                window_coord[d] = 0;
            else if (window_coord[d] > maxv)
                window_coord[d] = maxv;

            image_index += input->offset[d] * window_coord[d];
        }

        func((size_t)image_index, window_index);
    }
}

__global__ void erode_kernel(Image* input, Image* output, Window* window)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;

    uint8_t pmin = 255;
    window_map(input, window, i, [&](auto image_index, auto _) {
        uint8_t v = input->data[image_index];
        if (v < pmin)
            pmin = v;
    });

    output->data[i] = pmin;
}

__global__ void convolve_kernel(Image* input, Image* output, Window* window)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;

    float result = 0.0f;
    window_map(input, window, i, [&](auto image_index, auto window_index) {
        result += input->data[image_index] * window->data[window_index];
    });

    output->data[i] = (uint8_t)result;
}

DeviceImage* image_similar_device_from_host(Image* image)
{
    auto d_image = new DeviceImage();
    auto tmp_image = Image();
    cudaMalloc(&d_image->self, sizeof(Image));

    cudaMalloc(&d_image->data, image->size);
    tmp_image.data = d_image->data;

    cudaMalloc(&d_image->shape, image->dimensions * sizeof(int));
    cudaMemcpy(d_image->shape, image->shape, image->dimensions * sizeof(int), cudaMemcpyHostToDevice);
    tmp_image.shape = d_image->shape;

    cudaMalloc(&d_image->offset, image->dimensions * sizeof(int));
    cudaMemcpy(d_image->offset, image->offset, image->dimensions * sizeof(int), cudaMemcpyHostToDevice);
    tmp_image.offset = d_image->offset;

    d_image->dimensions = image->dimensions;
    tmp_image.dimensions = d_image->dimensions;
    d_image->size = image->size;
    tmp_image.size = d_image->size;

    cudaMemcpy(d_image->self, &tmp_image, sizeof(Image), cudaMemcpyHostToDevice);

    return d_image;
}

DeviceImage* image_device_from_host(Image* image)
{
    auto d_image = image_similar_device_from_host(image);

    cudaMemcpy(d_image->data, image->data, image->size, cudaMemcpyHostToDevice);

    return d_image;
}

DeviceImage* image_device_convert_from_host(Image* image)
{
    auto d_image = image_device_from_host(image);

    image_destroy(image);

    return d_image;
}

void image_destroy_device(DeviceImage* d_image)
{
    cudaFree(d_image->data);
    cudaFree(d_image->shape);
    cudaFree(d_image->offset);
    cudaFree(d_image->self);
    delete d_image;
}

DeviceWindow* window_similar_device_from_host(Window* window)
{
    auto d_window = new DeviceWindow();
    auto tmp_window = Window();
    cudaMalloc(&d_window->self, sizeof(Window));

    cudaMalloc(&d_window->data, window->size * sizeof(float));
    tmp_window.data = d_window->data;

    cudaMalloc(&d_window->shape, window->dimensions * sizeof(int));
    cudaMemcpy(d_window->shape, window->shape, window->dimensions * sizeof(int), cudaMemcpyHostToDevice);
    tmp_window.shape = d_window->shape;

    cudaMalloc(&d_window->offset, window->dimensions * sizeof(int));
    cudaMemcpy(d_window->offset, window->offset, window->dimensions * sizeof(int), cudaMemcpyHostToDevice);
    tmp_window.offset = d_window->offset;

    d_window->dimensions = window->dimensions;
    tmp_window.dimensions = d_window->dimensions;
    d_window->size = window->size;
    tmp_window.size = d_window->size;

    cudaMemcpy(d_window->self, &tmp_window, sizeof(Window), cudaMemcpyHostToDevice);

    return d_window;
}

DeviceWindow* window_device_from_host(Window* window)
{
    auto d_window = window_similar_device_from_host(window);

    cudaMemcpy(d_window->data, window->data, window->size * sizeof(float), cudaMemcpyHostToDevice);

    return d_window;
}

DeviceWindow* window_device_convert_from_host(Window* window)
{
    auto d_window = window_device_from_host(window);

    window_destroy(window);

    return d_window;
}

void window_destroy_device(DeviceWindow* d_window)
{
    cudaFree(d_window->data);
    cudaFree(d_window->shape);
    cudaFree(d_window->offset);
    cudaFree(d_window->self);
    delete d_window;
}

void benchmark(VglImage* vglimage, size_t rounds, std::function<void(VglImage*, std::string)> save_image)
{
    auto image = image_from_vglimage(vglimage);
    auto dimensions = image->dimensions;

    auto const THREADS_PER_BLOCK = 256;
    auto const BLOCKS = (int)((image->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    auto d_input = image_device_from_host(image);
    auto d_output = image_similar_device_from_host(image);
    auto d_temp = image_similar_device_from_host(image);

    auto d_cross_window = window_device_convert_from_host(window_create_from_type(WindowType::CROSS, dimensions));
    auto d_cube_window = window_device_convert_from_host(window_create_from_type(WindowType::CUBE, dimensions));
    auto d_mean_window = window_device_convert_from_host(window_create_from_type(WindowType::MEAN, dimensions));

    auto d_cube_window_array = new DeviceWindow*[dimensions + 1];
    auto d_mean_window_array = new DeviceWindow*[dimensions + 1];
    {
        auto cube_window_1d = window_create_from_type(WindowType::CUBE, 1);
        auto mean_window_1d = window_create_from_type(WindowType::MEAN, 1);

        for (int i = 1; i <= dimensions; ++i) {
            cube_window_1d->shape[i] = 3;
            mean_window_1d->shape[i] = 3;

            d_cube_window_array[i] = window_device_from_host(cube_window_1d);
            d_mean_window_array[i] = window_device_from_host(mean_window_1d);

            cube_window_1d->shape[i] = 1;
            mean_window_1d->shape[i] = 1;
        }

        window_destroy(cube_window_1d);
        window_destroy(mean_window_1d);
    }

    auto save_sample = [&](std::string name) {
        cudaMemcpy(vglimage->getImageData(), d_output->data, image->size, cudaMemcpyDeviceToHost);
        save_image(vglimage, name);
    };

    auto builder = BenchmarkBuilder();
    builder.attach({
        .name = "upload",
        .type = "group",
        .group = "memory",
        .func = [&] { cudaMemcpy(d_input->data, image->data, image->size, cudaMemcpyHostToDevice); },
    });
    builder.attach({
        .name = "download",
        .type = "group",
        .group = "memory",
        .func = [&] { cudaMemcpy(image->data, d_output->data, image->size, cudaMemcpyDeviceToHost); },
    });
    builder.attach({
        .name = "copy",
        .type = "group",
        .group = "memory",
        .post = save_sample,
        .func = [&] { cudaMemcpy(d_output->data, d_input->data, image->size, cudaMemcpyDeviceToDevice); },
    });
    builder.attach({
        .name = "invert",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] {
            invert_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_input->self);
            cudaDeviceSynchronize();
        },
    });
    builder.attach({
        .name = "threshold",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] {
            threshold_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_output->self, 128, 255);
            cudaDeviceSynchronize();
        },
    });
    builder.attach({
        .name = "erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            erode_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_output->self, d_cube_window->self);
            cudaDeviceSynchronize();
        },
    });
    builder.attach({
        .name = "split-erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            erode_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_temp->self, d_cube_window_array[1]->self);
            cudaDeviceSynchronize();
            for (int i = 2; i <= dimensions; ++i) {
                if (i & 0b1) {
                    erode_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_output->self, d_temp->self, d_cube_window_array[i]->self);
                } else {
                    erode_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_temp->self, d_output->self, d_cube_window_array[i]->self);
                }
                cudaDeviceSynchronize();
            }
            if (dimensions & 0b1) {
                cudaMemcpy(d_output->data, d_temp->data, image->size, cudaMemcpyDeviceToDevice);
            }
        },
    });
    builder.attach({
        .name = "erode-cross",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            erode_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_output->self, d_cross_window->self);
            cudaDeviceSynchronize();
        },
    });
    builder.attach({
        .name = "convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            convolve_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_output->self, d_mean_window->self);
            cudaDeviceSynchronize();
        },
    });
    builder.attach({
        .name = "split-convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            convolve_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input->self, d_temp->self, d_mean_window_array[1]->self);
            cudaDeviceSynchronize();
            for (int i = 2; i <= dimensions; ++i) {
                if (i & 0b1)
                    convolve_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_output->self, d_temp->self, d_mean_window_array[i]->self);
                else
                    convolve_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_temp->self, d_output->self, d_mean_window_array[i]->self);
                cudaDeviceSynchronize();
            }
            if (dimensions & 0b1) {
                cudaMemcpy(d_output->data, d_temp->data, image->size, cudaMemcpyDeviceToDevice);
            }
        },
    });
    builder.run(rounds);

    image_destroy(image);
    image_destroy_device(d_input);
    image_destroy_device(d_output);
    image_destroy_device(d_temp);
    window_destroy_device(d_cross_window);
    window_destroy_device(d_cube_window);
    window_destroy_device(d_mean_window);
    for (auto i = 1; i <= dimensions; ++i) {
        window_destroy_device(d_cube_window_array[i]);
        window_destroy_device(d_mean_window_array[i]);
    }
    delete[] d_cube_window_array;
    delete[] d_mean_window_array;
}
