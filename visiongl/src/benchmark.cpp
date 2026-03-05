#include <visiongl/cl/cl2cpp_ND.hpp>
#include <visiongl/cl/cl2cpp_shaders.hpp>
#include <visiongl/cl/image.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/context.hpp>
#include <visiongl/image.hpp>
#include <visiongl/shape.hpp>
#include <visiongl/strel.hpp>

#include <utils.hpp>

#ifndef FORCE_BUFFER
#    define FORCE_BUFFER (true)
#endif

void benchmark_nd(VglImage* input, size_t rounds, std::function<void(VglImage*, std::string)> save_image)
{
    vglClInit();
    vglClForceAsBuf(input);

    auto dimensions = input->ndim;
    auto output = vglCreateImage(input);
    auto tmp = vglCreateImage(input);

    auto strel_cross = VglStrEl(VGL_STREL_CROSS, dimensions);
    auto strel_cube = VglStrEl(VGL_STREL_CUBE, dimensions);
    auto strel_mean = VglStrEl(VGL_STREL_MEAN, dimensions);
    VglStrEl* strel_cube_array[VGL_ARR_SHAPE_SIZE];
    VglStrEl* strel_mean_array[VGL_ARR_SHAPE_SIZE];

    int strelShape[VGL_ARR_SHAPE_SIZE];
    for (int i = 0; i < VGL_ARR_SHAPE_SIZE; i++)
        strelShape[i] = 1;

    float data_cube[3] = { 1.0f, 1.0f, 1.0f };
    float data_mean[3] = { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f };
    for (int i = 1; i <= dimensions; ++i) {
        strelShape[i] = 3;
        VglShape* tmpShape = new VglShape(strelShape, dimensions);
        strel_cube_array[i] = new VglStrEl(data_cube, tmpShape);
        strel_mean_array[i] = new VglStrEl(data_mean, tmpShape);
        delete tmpShape;
        strelShape[i] = 1;
    }

    auto save_sample = [&](std::string name) {
        save_image(output, name);
    };

    auto builder = BenchmarkBuilder();
    builder.attach({
        .name = "upload",
        .type = "group",
        .group = "memory",
        .func = [&] {
            vglSetContext(input, VGL_RAM_CONTEXT);
            vglClUpload(input);
        },
    });
    builder.attach({
        .name = "download",
        .type = "group",
        .group = "memory",
        .func = [&] {
            vglSetContext(input, VGL_CL_CONTEXT);
            vglClDownload(input);
        },
    });
    builder.attach({
        .name = "copy",
        .type = "group",
        .group = "memory",
        .post = save_sample,
        .func = [&] {
            vglClNdCopy(input, output);
        },
    });
    builder.attach({
        .name = "invert",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { vglClNdNot(input, output); },
    });
    builder.attach({
        .name = "threshold",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { vglClNdThreshold(input, output, 128, 255); },
    });
    builder.attach({
        .name = "erode-cross",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglClNdErode(input, output, &strel_cross); },
    });
    builder.attach({
        .name = "erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglClNdErode(input, output, &strel_cube); },
    });
    builder.attach({
        .name = "split-erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            vglClNdErode(input, tmp, strel_cube_array[1]);
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    vglClNdErode(output, tmp, strel_cube_array[i]);
                else
                    vglClNdErode(tmp, output, strel_cube_array[i]);
            if (dimensions & 0b1)
                vglClNdCopy(tmp, output);
        },
    });
    builder.attach({
        .name = "convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglClNdConvolution(input, output, &strel_mean); },
    });
    builder.attach({
        .name = "split-convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            vglClNdConvolution(input, tmp, strel_mean_array[1]);
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    vglClNdConvolution(output, tmp, strel_mean_array[i]);
                else
                    vglClNdConvolution(tmp, output, strel_mean_array[i]);
            if (dimensions & 0b1)
                vglClNdCopy(tmp, output);
        },
    });
    builder.run(rounds);

    delete output;
    delete tmp;
    for (int i = 1; i <= dimensions; ++i) {
        delete strel_cube_array[i];
        delete strel_mean_array[i];
    }
}

void benchmark_2d(VglImage* input, size_t rounds, std::function<void(VglImage*, std::string)> save_image)
{
    vglClInit();

    auto dimensions = input->ndim;
    auto output = vglCreateImage(input);
    auto tmp = vglCreateImage(input);
    auto strel_cross = VglStrEl(VGL_STREL_CROSS, dimensions);
    auto strel_cube = VglStrEl(VGL_STREL_CUBE, dimensions);
    auto strel_mean = VglStrEl(VGL_STREL_MEAN, dimensions);
    auto strel_cube_1d = VglStrEl(VGL_STREL_CUBE, 1);
    auto strel_mean_1d = VglStrEl(VGL_STREL_MEAN, 1);

    auto save_sample = [&](std::string name) {
        save_image(output, name);
    };

    auto builder = BenchmarkBuilder();
    builder.attach({
        .name = "upload",
        .type = "group",
        .group = "memory",
        .func = [&] {
            vglSetContext(input, VGL_RAM_CONTEXT);
            vglClUpload(input);
        },
    });
    builder.attach({
        .name = "download",
        .type = "group",
        .group = "memory",
        .func = [&] {
            vglSetContext(input, VGL_CL_CONTEXT);
            vglClDownload(input);
        },
    });
    builder.attach({
        .name = "copy",
        .type = "group",
        .group = "memory",
        .post = save_sample,
        .func = [&] {
            vglClCopy(input, output);
        },
    });
    builder.attach({
        .name = "invert",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { vglClInvert(input, output); },
    });
    builder.attach({
        .name = "threshold",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { vglClThreshold(input, output, 0.5, 1); },
    });
    builder.attach({
        .name = "erode-cross",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglClErode(input, output, strel_cross.getData(), strel_cross.getShape()[VGL_SHAPE_WIDTH], strel_cross.getShape()[VGL_SHAPE_HEIGHT]); },
    });
    builder.attach({
        .name = "erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglClErode(input, output, strel_cube.getData(), strel_cube.getShape()[VGL_SHAPE_WIDTH], strel_cube.getShape()[VGL_SHAPE_HEIGHT]); },
    });
    builder.attach({
        .name = "split-erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            vglClErode(input, tmp, strel_cube_1d.getData(), strel_cube_1d.getShape()[VGL_SHAPE_WIDTH], strel_cube_1d.getShape()[VGL_SHAPE_HEIGHT]);
            vglClErode(tmp, output, strel_cube_1d.getData(), strel_cube_1d.getShape()[VGL_SHAPE_HEIGHT], strel_cube_1d.getShape()[VGL_SHAPE_WIDTH]);
        },
    });
    builder.attach({
        .name = "convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglClConvolution(input, output, strel_mean.getData(), strel_mean.getShape()[VGL_SHAPE_WIDTH], strel_mean.getShape()[VGL_SHAPE_HEIGHT]); },
    });
    builder.attach({
        .name = "split-convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            vglClConvolution(input, tmp, strel_mean_1d.getData(), strel_mean_1d.getShape()[VGL_SHAPE_WIDTH], strel_mean_1d.getShape()[VGL_SHAPE_HEIGHT]);
            vglClConvolution(tmp, output, strel_mean_1d.getData(), strel_mean_1d.getShape()[VGL_SHAPE_HEIGHT], strel_mean_1d.getShape()[VGL_SHAPE_WIDTH]);
        },
    });
    builder.run(rounds);

    delete output;
    delete tmp;
}

void benchmark_3d(VglImage* input, size_t rounds, std::function<void(VglImage*, std::string)> save_image)
{
    vglClInit();

    auto dimensions = input->ndim;
    auto output = vglCreateImage(input);
    auto tmp = vglCreateImage(input);
    auto strel_cross = VglStrEl(VGL_STREL_CROSS, dimensions);
    auto strel_cube = VglStrEl(VGL_STREL_CUBE, dimensions);
    auto strel_mean = VglStrEl(VGL_STREL_MEAN, dimensions);
    auto strel_cube_1d = VglStrEl(VGL_STREL_CUBE, 1);
    auto strel_mean_1d = VglStrEl(VGL_STREL_MEAN, 1);

    auto save_sample = [&](std::string name) {
        save_image(output, name);
    };

    auto builder = BenchmarkBuilder();
    builder.attach({
        .name = "upload",
        .type = "group",
        .group = "memory",
        .func = [&] {
            vglSetContext(input, VGL_RAM_CONTEXT);
            vglClUpload(input);
        },
    });
    builder.attach({
        .name = "download",
        .type = "group",
        .group = "memory",
        .func = [&] {
            vglSetContext(input, VGL_CL_CONTEXT);
            vglClDownload(input);
        },
    });
    builder.attach({
        .name = "copy",
        .type = "group",
        .group = "memory",
        .post = save_sample,
        .func = [&] {
            vglCl3dCopy(input, output);
        },
    });
    builder.attach({
        .name = "invert",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { vglCl3dNot(input, output); },
    });
    builder.attach({
        .name = "threshold",
        .type = "group",
        .group = "point",
        .post = save_sample,
        .func = [&] { vglCl3dThreshold(input, output, 0.5, 1); },
    });
    builder.attach({
        .name = "erode-cross",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglCl3dErode(input, output, strel_cross.getData(), strel_cross.getShape()[VGL_SHAPE_D1], strel_cross.getShape()[VGL_SHAPE_D2], strel_cross.getShape()[VGL_SHAPE_D3]); },
    });
    builder.attach({
        .name = "erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglCl3dErode(input, output, strel_cube.getData(), strel_cube.getShape()[VGL_SHAPE_D1], strel_cube.getShape()[VGL_SHAPE_D2], strel_cube.getShape()[VGL_SHAPE_D3]); },
    });
    builder.attach({
        .name = "split-erode-cube",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            vglCl3dErode(input, output, strel_cube_1d.getData(), strel_cube_1d.getShape()[VGL_SHAPE_D1], strel_cube_1d.getShape()[VGL_SHAPE_D2], strel_cube_1d.getShape()[VGL_SHAPE_D3]);
            vglCl3dErode(output, tmp, strel_cube_1d.getData(), strel_cube_1d.getShape()[VGL_SHAPE_D2], strel_cube_1d.getShape()[VGL_SHAPE_D1], strel_cube_1d.getShape()[VGL_SHAPE_D3]);
            vglCl3dErode(tmp, output, strel_cube_1d.getData(), strel_cube_1d.getShape()[VGL_SHAPE_D3], strel_cube_1d.getShape()[VGL_SHAPE_D2], strel_cube_1d.getShape()[VGL_SHAPE_D1]);
        },
    });
    builder.attach({
        .name = "convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] { vglCl3dConvolution(input, output, strel_mean.getData(), strel_mean.getShape()[VGL_SHAPE_D1], strel_mean.getShape()[VGL_SHAPE_D2], strel_mean.getShape()[VGL_SHAPE_D3]); },
    });
    builder.attach({
        .name = "split-convolve",
        .type = "single",
        .post = save_sample,
        .func = [&] {
            vglCl3dConvolution(input, output, strel_mean_1d.getData(), strel_mean_1d.getShape()[VGL_SHAPE_D1], strel_mean_1d.getShape()[VGL_SHAPE_D2], strel_mean_1d.getShape()[VGL_SHAPE_D3]);
            vglCl3dConvolution(output, tmp, strel_mean_1d.getData(), strel_mean_1d.getShape()[VGL_SHAPE_D2], strel_mean_1d.getShape()[VGL_SHAPE_D1], strel_mean_1d.getShape()[VGL_SHAPE_D3]);
            vglCl3dConvolution(tmp, output, strel_mean_1d.getData(), strel_mean_1d.getShape()[VGL_SHAPE_D3], strel_mean_1d.getShape()[VGL_SHAPE_D2], strel_mean_1d.getShape()[VGL_SHAPE_D1]);
        },
    });
    builder.run(rounds);

    delete output;
    delete tmp;
}

void benchmark(VglImage* image, size_t rounds, std::function<void(VglImage*, std::string)> save_image)
{
    if (FORCE_BUFFER || image->ndim < 2 || image->ndim > 3) {
        benchmark_nd(image, rounds, save_image);
    } else if (image->ndim == 2) {
        benchmark_2d(image, rounds, save_image);
    } else /* (image->ndim == 3) */ {
        benchmark_3d(image, rounds, save_image);
    }
}
