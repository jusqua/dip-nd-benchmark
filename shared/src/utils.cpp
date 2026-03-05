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

void BenchmarkBuilder::perform_benchmark(std::size_t rounds, BenchmarkSpec const& spec)
{
    auto time_start_once = std::chrono::high_resolution_clock::now();
    spec.func();
    auto time_end_once = std::chrono::high_resolution_clock::now();
    double once_duration = std::chrono::duration<double>(time_end_once - time_start_once).count();

    std::cout << spec.name << "," << spec.type << "," << spec.group << "," << once_duration;

    if (rounds <= 1) {
        std::cout << "\n";
        return;
    }

    auto mean_start_times = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < rounds; ++i)
        spec.func();
    auto mean_end_times = std::chrono::high_resolution_clock::now();

    double mean_duration = std::chrono::duration<double>(mean_end_times - mean_start_times).count() / rounds;
    std::cout << "," << mean_duration << "\n";
}

void BenchmarkBuilder::attach(BenchmarkSpec&& spec)
{
    m_specs.emplace_back(spec);
}

void BenchmarkBuilder::run(std::size_t rounds)
{
    std::cout << "operator,type,group,once";
    if (rounds <= 1)
        std::cout << "\n";
    else
        std::cout << ",mean\n";

    for (auto const& spec : m_specs) {
        perform_benchmark(rounds, spec);
        if (spec.post != nullptr)
            spec.post(spec.name);
    }
}
