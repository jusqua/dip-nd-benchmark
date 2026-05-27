#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <visiongl/image.hpp>
#include <visiongl/shape.hpp>

#include <utils.hpp>
#include <visiongl/strel.hpp>

Image* image_from_vglimage(VglImage* vglimage)
{
    auto image = new Image();

    image->data = new uint8_t[vglimage->vglShape->getSize()];
    image->shape = new int[vglimage->vglShape->getNdim() + 1];
    image->offset = new int[vglimage->vglShape->getNdim() + 1];
    image->dimensions = vglimage->vglShape->getNdim();
    image->size = vglimage->vglShape->getSize();

    std::copy_n(vglimage->getImageData(), image->size, image->data);
    std::copy_n(vglimage->vglShape->getShape(), image->dimensions + 1, image->shape);
    std::copy_n(vglimage->vglShape->getOffset(), image->dimensions + 1, image->offset);

    return image;
}

Image* image_convert_from_vglimage(VglImage* vglimage)
{
    auto image = image_from_vglimage(vglimage);

    delete vglimage;

    return image;
}

void image_destroy(Image* image)
{
    delete[] image->data;
    delete[] image->shape;
    delete[] image->offset;
    delete image;
}

Window* window_from_vglstrel(VglStrEl* vglstrel)
{
    auto window = new Window();

    window->data = new float[vglstrel->vglShape->getSize()];
    window->shape = new int[vglstrel->vglShape->getNdim() + 1];
    window->offset = new int[vglstrel->vglShape->getNdim() + 1];
    window->dimensions = vglstrel->vglShape->getNdim();
    window->size = vglstrel->vglShape->getSize();

    std::copy_n(vglstrel->getData(), window->size, window->data);
    std::copy_n(vglstrel->vglShape->getShape(), window->dimensions + 1, window->shape);
    std::copy_n(vglstrel->vglShape->getOffset(), window->dimensions + 1, window->offset);

    return window;
}

Window* window_convert_from_vglstrel(VglStrEl* vglstrel)
{
    auto window = window_from_vglstrel(vglstrel);

    delete vglstrel;

    return window;
}

void window_destroy(Window* window)
{
    delete[] window->data;
    delete[] window->shape;
    delete[] window->offset;
    delete window;
}

Window* window_create_from_type(WindowType type, uint8_t dimension)
{
    switch (type) {
    case WindowType::CROSS:
        return window_from_vglstrel(new VglStrEl(VGL_STREL_CROSS, dimension));
    case WindowType::CUBE:
        return window_from_vglstrel(new VglStrEl(VGL_STREL_CUBE, dimension));
    case WindowType::MEAN:
        return window_from_vglstrel(new VglStrEl(VGL_STREL_MEAN, dimension));
    default:
        return new Window();
    }
}

void BenchmarkBuilder::perform_benchmark(std::size_t rounds, BenchmarkSpec const& spec)
{
    // Warm up
    spec.func();

    for (size_t i = 0; i < rounds; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        spec.func();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout
            << spec.name << ","
            << spec.type << ","
            << spec.group << ","
            << std::chrono::duration<double>(end - start).count() << "\n";
    }

    if (spec.post != nullptr)
        spec.post(spec.name);
}

void BenchmarkBuilder::attach(BenchmarkSpec&& spec)
{
    m_specs.emplace_back(spec);
}

void BenchmarkBuilder::run(std::size_t rounds)
{
    if (rounds < 1) rounds = 1;

    std::cout << "operator,type,group,duration\n";
    for (auto const& spec : m_specs) perform_benchmark(rounds, spec);
}
