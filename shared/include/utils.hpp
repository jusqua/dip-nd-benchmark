#ifndef DIP_ND_BENCHMARK_UTILS_HPP
#define DIP_ND_BENCHMARK_UTILS_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <visiongl/image.hpp>
#include <visiongl/strel.hpp>

struct Image {
    uint8_t* data;
    int* shape;
    int* offset;
    uint8_t dimensions;
    size_t size;
};

struct DeviceImage : Image {
    Image* self;
};

Image* image_from_vglimage(VglImage* vglimage);
Image* image_convert_from_vglimage(VglImage* vglimage);
void image_destroy(Image* image);

struct Window {
    float* data;
    int* shape;
    int* offset;
    uint8_t dimensions;
    size_t size;
};

struct DeviceWindow : Window {
    Window* self;
};

enum class WindowType {
    CUBE,
    MEAN,
    CROSS
};

Window* window_from_vglstrel(VglStrEl* vglstrel);
Window* window_convert_from_vglstrel(VglStrEl* vglstrel);
void window_destroy(Window* window);
Window* window_create_from_type(WindowType type, uint8_t dimension);

struct BenchmarkSpec {
    std::string name;
    std::string type;
    std::string group = "";
    std::function<void(std::string)> post = nullptr;
    std::function<void(void)> func;
};

class BenchmarkBuilder {
private:
    std::vector<BenchmarkSpec> m_specs;

    void perform_benchmark(std::size_t rounds, BenchmarkSpec const& spec);

public:
    void attach(BenchmarkSpec&& spec);
    void run(std::size_t rounds);
};

void benchmark(VglImage* vglimage, size_t rounds, std::function<void(VglImage*, std::string)> save_image);

#endif // DIP_ND_BENCHMARK_UTILS_HPP
