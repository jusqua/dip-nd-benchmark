#include <cstring>
#include <filesystem>

#include <visiongl/context.hpp>
#include <visiongl/image.hpp>
#include <visiongl/shape.hpp>

#include <utils.hpp>

int main(int argc, char** argv)
{
    auto usage = "Usage: benchmark <input pattern> <index 0> <index n> <rounds> <output folder> [<d1> <d2> ... <dN>]\n"
                 "This program reads a stack of image files and saves the results after benchmarking some operations.\n"
                 "Both input and output files require a printf-like integer format specifier (%d) which will be replaced by the integers from index 0 to index N\n"
                 "Optionally it's possible to define an alternative geometry to the image by adding the dimension sizes after the output folder as follows.\n";

    constexpr int ARGD1 = 6;

    if (argc < ARGD1) {
        printf("%s", usage);
        exit(EXIT_FAILURE);
    }

    char* inpath = argv[1];
    int i0 = atoi(argv[2]);
    int iN = atoi(argv[3]);
    int rounds = atoi(argv[4]);
    char* outpath = argv[5];

    int shape[VGL_ARR_SHAPE_SIZE] = { 0 };
    int ndim = 3;
    if (argc < ARGD1) {
        shape[VGL_SHAPE_D3] = iN - i0 + 1;
    } else {
        ndim = argc - ARGD1;
        for (int i = 0; i < ndim; i++) {
            shape[1 + i] = atoi(argv[ARGD1 + i]);
        }
    }

    char* tmppath = (char*)malloc(strlen(inpath) + 256);
    sprintf(tmppath, inpath, i0);
    VglImage* firstImage = vglLoadImage(tmppath);
    int baseShape[VGL_ARR_SHAPE_SIZE] = { 0 };
    baseShape[VGL_SHAPE_NCHANNELS] = firstImage->getNChannels();
    baseShape[VGL_SHAPE_WIDTH] = firstImage->getWidth();
    baseShape[VGL_SHAPE_HEIGHT] = firstImage->getHeight();
    baseShape[VGL_SHAPE_LENGTH] = iN - i0 + 1;
    VglShape* originalVglShape = new VglShape(baseShape, 3);

    VglImage* input = vglLoadNdImage(inpath, i0, iN, shape, ndim);

    benchmark(input, rounds, [&](VglImage* image, char const* operation) {
        vglCheckContext(image, VGL_RAM_CONTEXT);
        if (ndim <= 2)
            vglReshape(image, originalVglShape);

        auto outfilename = (char*)malloc(strlen(outpath) + 255);
        sprintf(outfilename, "%s/%s", outpath, operation);

        if (!std::filesystem::exists(outfilename))
            std::filesystem::create_directories(outfilename);

        sprintf(outfilename, "%s/%%05d.tif", outfilename);
        vglSaveNdImage(outfilename, image, i0);

        free(outfilename);
    });

    delete originalVglShape;
    delete firstImage;
    delete input;
    free(tmppath);

    return 0;
}
