#include <functional>
#include <visiongl/image.hpp>

void benchmark(VglImage *image, size_t rounds, std::function<void(VglImage *, const char *)> save_image);
