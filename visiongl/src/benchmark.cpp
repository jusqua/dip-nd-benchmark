#include <visiongl/cl/cl2cpp_ND.hpp>
#include <visiongl/cl/image.hpp>
#include <visiongl/constants.hpp>
#include <visiongl/context.hpp>
#include <visiongl/shape.hpp>
#include <visiongl/strel.hpp>

#include <utils.hpp>

void benchmark(VglImage* image, size_t rounds, std::function<void(VglImage*, char const*)> save_image)
{
    vglClInit();
    vglClForceAsBuf(image);

    auto dimensions = image->ndim;
    VglImage* output = vglCreateImage(image);
    VglImage* tmp = vglCreateImage(image);

    VglStrEl* strel_cross = new VglStrEl(VGL_STREL_CROSS, dimensions);
    VglStrEl* strel_cube = new VglStrEl(VGL_STREL_CUBE, dimensions);
    VglStrEl* strel_mean = new VglStrEl(VGL_STREL_MEAN, dimensions);
    VglStrEl* strel_cube_array[VGL_ARR_SHAPE_SIZE];
    VglStrEl* strel_mean_array[VGL_ARR_SHAPE_SIZE];

    int strelShape[VGL_ARR_SHAPE_SIZE];
    for (int i = 0; i < VGL_ARR_SHAPE_SIZE; i++) {
        strelShape[i] = 1;
    }

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
        save_image(output, name.c_str());
    };

    auto builder = BenchmarkBuilder();

    builder.attach({
        .name = "upload",
        .func = [&] {
            vglSetContext(image, VGL_RAM_CONTEXT);
            vglClUpload(image);
        },
    });
    builder.attach({
        .name = "download",
        .func = [&] {
            vglSetContext(image, VGL_CL_CONTEXT);
            vglClDownload(image);
        },
    });
    builder.attach({ .name = "copy",
        .func = [&] {
            vglClNdCopy(image, output);
        },
        .post = save_sample });
    builder.attach({ .name = "invert",
        .func = [&] { vglClNdNot(image, output); },
        .post = save_sample });
    builder.attach({ .name = "threshold",
        .func = [&] { vglClNdThreshold(image, output, 128, 255); },
        .post = save_sample });
    builder.attach({ .name = "erode-cube",
        .func = [&] { vglClNdErode(image, output, strel_cube); },
        .post = save_sample });
    builder.attach({ .name = "split-erode-cube",
        .func = [&] {
            vglClNdErode(image, tmp, strel_cube_array[1]);
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    vglClNdErode(output, tmp, strel_cube_array[i]);
                else
                    vglClNdErode(tmp, output, strel_cube_array[i]);
            if (dimensions & 0b1)
                vglClNdCopy(tmp, output);
        },
        .post = save_sample });
    builder.attach({ .name = "erode-cross",
        .func = [&] { vglClNdErode(image, output, strel_cross); },
        .post = save_sample });
    builder.attach({ .name = "convolve",
        .func = [&] { vglClNdConvolution(image, output, strel_mean); },
        .post = save_sample });
    builder.attach({ .name = "split-convolve",
        .func = [&] {
            vglClNdConvolution(image, tmp, strel_mean_array[1]);
            for (int i = 2; i <= dimensions; ++i)
                if (i & 0b1)
                    vglClNdConvolution(output, tmp, strel_mean_array[i]);
                else
                    vglClNdConvolution(tmp, output, strel_mean_array[i]);
            if (dimensions & 0b1)
                vglClNdCopy(tmp, output);
        },
        .post = save_sample });

    builder.run(rounds);

    delete output;
    delete tmp;
    delete strel_cross;
    delete strel_cube;
    delete strel_mean;

    for (int i = 1; i <= dimensions; ++i) {
        delete strel_cube_array[i];
        delete strel_mean_array[i];
    }
}
