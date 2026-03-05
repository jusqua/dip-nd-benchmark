#include <cstring>
#include <filesystem>
#include <iostream>

#include <visiongl/context.hpp>
#include <visiongl/image.hpp>
#include <visiongl/shape.hpp>

#include <utils.hpp>

int main(int argc, char** argv)
{
    auto usage = "Usage: benchmark <input pattern> <index 0> <index n> <rounds> <output folder> [<d1> <d2> ... <dN>]\n";

    constexpr int ARGD1 = 6;

    if (argc < ARGD1) {
        std::cout << usage << "\n";
        std::exit(EXIT_FAILURE);
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

    auto tmppath = new char[strlen(outpath) + 256];
    sprintf(tmppath, inpath, i0);
    auto firstImage = vglLoadImage(tmppath);
    int baseShape[VGL_ARR_SHAPE_SIZE] = { 0 };
    baseShape[VGL_SHAPE_NCHANNELS] = firstImage->getNChannels();
    baseShape[VGL_SHAPE_WIDTH] = firstImage->getWidth();
    baseShape[VGL_SHAPE_HEIGHT] = firstImage->getHeight();
    baseShape[VGL_SHAPE_LENGTH] = iN - i0 + 1;
    delete[] tmppath;
    delete firstImage;

    auto vglshape = new VglShape(baseShape, 3);
    auto vglimage = vglLoadNdImage(inpath, i0, iN, shape, ndim);

    benchmark(vglimage, rounds, [&](VglImage* output, std::string codename) {
        vglCheckContext(output, VGL_RAM_CONTEXT);
        if (ndim <= 2)
            vglReshape(output, vglshape);

        auto outfilename = new char[strlen(outpath) + 256];
        sprintf(outfilename, "%s/%s", outpath, codename.c_str());

        if (!std::filesystem::exists(outfilename))
            std::filesystem::create_directories(outfilename);

        sprintf(outfilename, "%s/%%05d.tif", outfilename);
        vglSaveNdImage(outfilename, output, i0);

        delete[] outfilename;
    });

    delete vglimage;
    delete vglshape;

    return 0;
}
