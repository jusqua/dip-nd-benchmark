#include <cuda_runtime.h>

#include <visiongl/constants.hpp>
#include <visiongl/image.hpp>
#include <visiongl/strel.hpp>

#include <utils.hpp>

__global__ void invertKernel(Image const* input, Image* output)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;
    output->data[i] = 255 - input->data[i];
}

__global__ void thresholdKernel(Image const* input, Image* output, uint8_t threshold, uint8_t max_value)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;
    output->data[i] = input->data[i] > threshold ? max_value : 0;
}

template<typename Func = std::function<void(size_t, size_t)>>
__device__ void windowMap(Image* input, Window* window, size_t const index, Func&& func)
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

__global__ void erodeKernel(Image* input, Image* output, Window* window)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;

    uint8_t pmin = 255;
    windowMap(input, window, i, [&](auto image_index, auto _) {
        uint8_t v = input->data[image_index];
        if (v < pmin)
            pmin = v;
    });

    output->data[i] = pmin;
}

__global__ void convolveKernel(Image* input, Image* output, Window* window)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input->size)
        return;

    float result = 0.0f;
    windowMap(input, window, i, [&](auto image_index, auto window_index) {
        result += input->data[image_index] * window->data[window_index];
    });

    output->data[i] = (uint8_t)result;
}

static inline Image* copyImageStructToDevice(uint8_t* d_data, int* d_shape, int* d_offset, uint8_t dimensions, size_t size)
{
    Image h = Image(d_data, d_shape, d_offset, dimensions, size);
    Image* d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(Image));
    cudaMemcpy(d_ptr, &h, sizeof(Image), cudaMemcpyHostToDevice);
    return d_ptr;
}

static inline Window* copyWindowStructToDevice(float* d_data, int* d_shape, int* d_offset, uint8_t dimensions, size_t size)
{
    Window h = Window(d_data, d_shape, d_offset, dimensions, size);
    Window* d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(Window));
    cudaMemcpy(d_ptr, &h, sizeof(Window), cudaMemcpyHostToDevice);
    return d_ptr;
}

void benchmark(VglImage* image, size_t rounds, std::function<void(VglImage*, char const*)> save_image)
{
    auto dimensions = image->ndim;
    VglImage* output = vglCreateImage(image);
    VglImage* tmp = vglCreateImage(image);

    VglStrEl* strel_cross = new VglStrEl(VGL_STREL_CROSS, dimensions);
    VglStrEl* strel_cube = new VglStrEl(VGL_STREL_CUBE, dimensions);
    VglStrEl* strel_mean = new VglStrEl(VGL_STREL_MEAN, dimensions);
    float data_cube[3] = { 1.0f, 1.0f, 1.0f };
    float data_mean[3] = { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f };
    size_t image_size = image->vglShape->size;
    size_t dims_bytes = sizeof(int) * (dimensions + 1);

    int* d_shape = nullptr;
    int* d_offset = nullptr;
    cudaMalloc(&d_shape, dims_bytes);
    cudaMalloc(&d_offset, dims_bytes);
    cudaMemcpy(d_shape, image->vglShape->shape, dims_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, image->vglShape->offset, dims_bytes, cudaMemcpyHostToDevice);

    uint8_t* d_input_data = nullptr;
    uint8_t* d_output_data = nullptr;
    uint8_t* d_aux_data = nullptr;
    cudaMalloc(&d_input_data, image_size);
    cudaMalloc(&d_output_data, image_size);
    cudaMalloc(&d_aux_data, image_size);

    cudaMemcpy(d_input_data, image->getImageData(), image_size, cudaMemcpyHostToDevice);

    Image* d_input_img = copyImageStructToDevice(d_input_data, d_shape, d_offset, dimensions, image_size);
    Image* d_output_img = copyImageStructToDevice(d_output_data, d_shape, d_offset, dimensions, image_size);
    Image* d_aux_img = copyImageStructToDevice(d_aux_data, d_shape, d_offset, dimensions, image_size);

    size_t window_elements = strel_cross->vglShape->size;
    float* d_window_cross_data = nullptr;
    float* d_window_cube_data = nullptr;
    float* d_window_mean_data = nullptr;
    cudaMalloc(&d_window_cross_data, window_elements * sizeof(float));
    cudaMalloc(&d_window_cube_data, window_elements * sizeof(float));
    cudaMalloc(&d_window_mean_data, window_elements * sizeof(float));

    int* d_window_shape = nullptr;
    int* d_window_offset = nullptr;
    cudaMalloc(&d_window_shape, dims_bytes);
    cudaMalloc(&d_window_offset, dims_bytes);

    cudaMemcpy(d_window_cross_data, strel_cross->data, window_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_window_cube_data, strel_cube->data, window_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_window_mean_data, strel_mean->data, window_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_window_shape, strel_mean->vglShape->shape, dims_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_window_offset, strel_mean->vglShape->offset, dims_bytes, cudaMemcpyHostToDevice);

    Window* d_window_cross = copyWindowStructToDevice(d_window_cross_data, d_window_shape, d_window_offset, dimensions, window_elements);
    Window* d_window_cube = copyWindowStructToDevice(d_window_cube_data, d_window_shape, d_window_offset, dimensions, window_elements);
    Window* d_window_mean = copyWindowStructToDevice(d_window_mean_data, d_window_shape, d_window_offset, dimensions, window_elements);

    std::array<Window*, VGL_ARR_SHAPE_SIZE> d_window_cube_array;
    std::array<Window*, VGL_ARR_SHAPE_SIZE> d_window_mean_array;
    size_t window_linear_elements = 3;
    float* d_window_cube_linear_data = nullptr;
    float* d_window_mean_linear_data = nullptr;
    cudaMalloc(&d_window_cube_linear_data, window_linear_elements * sizeof(float));
    cudaMalloc(&d_window_mean_linear_data, window_linear_elements * sizeof(float));
    cudaMemcpy(d_window_cube_linear_data, data_cube, window_linear_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_window_mean_linear_data, data_mean, window_linear_elements * sizeof(float), cudaMemcpyHostToDevice);

    int window_linear_shape[VGL_ARR_SHAPE_SIZE];
    for (int i = 0; i < VGL_ARR_SHAPE_SIZE; ++i)
        window_linear_shape[i] = 1;

    for (int i = 1; i <= dimensions; ++i) {
        window_linear_shape[i] = 3;
        auto vgl_shape = new VglShape(window_linear_shape, dimensions);

        int* d_wshape = nullptr;
        int* d_woffset = nullptr;
        cudaMalloc(&d_wshape, dims_bytes);
        cudaMalloc(&d_woffset, dims_bytes);
        cudaMemcpy(d_wshape, vgl_shape->shape, dims_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_woffset, vgl_shape->offset, dims_bytes, cudaMemcpyHostToDevice);

        d_window_cube_array[i] = copyWindowStructToDevice(d_window_cube_linear_data, d_wshape, d_woffset, dimensions, window_linear_elements);
        d_window_mean_array[i] = copyWindowStructToDevice(d_window_mean_linear_data, d_wshape, d_woffset, dimensions, window_linear_elements);

        delete vgl_shape;
        window_linear_shape[i] = 1;
    }

    auto save_sample = [&](std::string name) {
        cudaMemcpy(tmp->getImageData(), d_output_data, image_size, cudaMemcpyDeviceToHost);
        save_image(tmp, name.c_str());
    };

    int const threadsPerBlock = 256;
    int const blocks = (int)((image_size + threadsPerBlock - 1) / threadsPerBlock);

    auto builder = BenchmarkBuilder();
    builder.attach({ .name = "upload",
        .type = "group",
        .group = "memory",
        .func = [&] {
            cudaMemcpy(d_input_data, image->getImageData(), image_size, cudaMemcpyHostToDevice);
        } });
    builder.attach({ .name = "download",
        .type = "group",
        .group = "memory",
        .func = [&] {
            cudaMemcpy(tmp->getImageData(), d_input_data, image_size, cudaMemcpyDeviceToHost);
        } });
    builder.attach({ .name = "copy",
        .type = "group",
        .group = "memory",
        .func = [&] {
            cudaMemcpy(d_output_data, d_input_data, image_size, cudaMemcpyDeviceToDevice);
        },
        .post = save_sample });
    builder.attach({ .name = "invert",
        .type = "group",
        .group = "point",
        .func = [&] {
            invertKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_output_img);
            cudaDeviceSynchronize();
        },
        .post = save_sample });
    builder.attach({ .name = "threshold",
        .type = "group",
        .group = "point",
        .func = [&] {
            thresholdKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_output_img, 128, 255);
            cudaDeviceSynchronize();
        },
        .post = save_sample });
    builder.attach({ .name = "erode-cube",
        .type = "single",
        .func = [&] {
            erodeKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_output_img, d_window_cube);
            cudaDeviceSynchronize();
        },
        .post = save_sample });
    builder.attach({ .name = "split-erode-cube",
        .type = "single",
        .func = [&] {
            erodeKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_aux_img, d_window_cube_array[1]);
            cudaDeviceSynchronize();
            for (int i = 2; i <= dimensions; ++i) {
                if (i & 0b1) {
                    erodeKernel<<<blocks, threadsPerBlock>>>(d_output_img, d_aux_img, d_window_cube_array[i]);
                } else {
                    erodeKernel<<<blocks, threadsPerBlock>>>(d_aux_img, d_output_img, d_window_cube_array[i]);
                }
                cudaDeviceSynchronize();
            }
            if (dimensions & 0b1) {
                cudaMemcpy(d_output_data, d_aux_data, image_size, cudaMemcpyDeviceToDevice);
            }
        },
        .post = save_sample });
    builder.attach({ .name = "erode-cross",
        .type = "single",
        .func = [&] {
            erodeKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_output_img, d_window_cross);
            cudaDeviceSynchronize();
        },
        .post = save_sample });
    builder.attach({ .name = "convolve",
        .type = "single",
        .func = [&] {
            convolveKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_output_img, d_window_mean);
            cudaDeviceSynchronize();
        },
        .post = save_sample });
    builder.attach({ .name = "split-convolve",
        .type = "single",
        .func = [&] {
            convolveKernel<<<blocks, threadsPerBlock>>>(d_input_img, d_aux_img, d_window_mean_array[1]);
            cudaDeviceSynchronize();
            for (int i = 2; i <= dimensions; ++i) {
                if (i & 0b1)
                    convolveKernel<<<blocks, threadsPerBlock>>>(d_output_img, d_aux_img, d_window_mean_array[i]);
                else
                    convolveKernel<<<blocks, threadsPerBlock>>>(d_aux_img, d_output_img, d_window_mean_array[i]);
                cudaDeviceSynchronize();
            }
            if (dimensions & 0b1) {
                cudaMemcpy(d_output_data, d_aux_data, image_size, cudaMemcpyDeviceToDevice);
            }
        },
        .post = save_sample });

    builder.run(rounds);

    cudaFree(d_window_cross_data);
    cudaFree(d_window_cube_data);
    cudaFree(d_window_mean_data);
    cudaFree(d_window_shape);
    cudaFree(d_window_offset);

    cudaFree(d_shape);
    cudaFree(d_offset);
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    cudaFree(d_aux_data);
    cudaFree(d_window_cube_linear_data);
    cudaFree(d_window_mean_linear_data);

    cudaFree(d_input_img);
    cudaFree(d_output_img);
    cudaFree(d_aux_img);

    cudaFree(d_window_cross);
    cudaFree(d_window_cube);
    cudaFree(d_window_mean);
    for (int i = 1; i <= dimensions; ++i) {
        cudaFree(d_window_cube_array[i]);
        cudaFree(d_window_mean_array[i]);
    }

    delete strel_cross;
    delete strel_cube;
    delete strel_mean;
    delete output;
    delete tmp;
}
