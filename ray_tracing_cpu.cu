#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

// Vec3 class for vector operations
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    Vec3 normalize() const {
        float mag = sqrt(x * x + y * y + z * z);
        return *this / mag;
    }
};

// Sphere structure
struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    bool intersect(const Vec3& origin, const Vec3& dir, float& t) const {
        Vec3 oc = origin - center;
        float b = oc.dot(dir);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - c;
        if (discriminant < 0) return false;
        t = -b - sqrt(discriminant);
        return t > 0;
    }
};

// Save the image to a PPM file
void save_image(const Vec3* framebuffer, int width, int height, const char* filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.put(static_cast<unsigned char>(fminf(framebuffer[i].x, 1.0f) * 255));
        file.put(static_cast<unsigned char>(fminf(framebuffer[i].y, 1.0f) * 255));
        file.put(static_cast<unsigned char>(fminf(framebuffer[i].z, 1.0f) * 255));
    }
    file.close();
}

int main() {
    int width = 3840;
    int height = 2160;
    int num_pixels = width * height;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Vec3> framebuffer(num_pixels);

    Sphere spheres[] = {
        { Vec3(0, 0, -3), 1, Vec3(1, 0, 0) },
        { Vec3(2, 0, -4), 1, Vec3(0, 1, 0) },
        { Vec3(-2, 0, -4), 1, Vec3(0, 0, 1) }
    };
    int sphere_count = 3;

    // Initialize the random number generator
    std::mt19937 rng(12345); // Seed with a constant value for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Loop over all pixels
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec3 pixel_color(0, 0, 0);
            int samples = 4; // Number of samples per pixel

            for (int s = 0; s < samples; ++s) {
                // Random sampling u and v
                float u = (x + dist(rng)) / width;
                float v = (y + dist(rng)) / height;

                // Generate the ray
                Vec3 ray_origin(0, 0, 0);
                Vec3 ray_dir = Vec3(u - 0.5f, v - 0.5f, -1).normalize();

                // Ray-sphere intersection check
                Vec3 sample_color(0, 0, 0);
                float t_min = 1e20f;
                for (int i = 0; i < sphere_count; ++i) {
                    float t;
                    if (spheres[i].intersect(ray_origin, ray_dir, t) && t < t_min) {
                        t_min = t;
                        sample_color = spheres[i].color;
                    }
                }

                // Accumulate the sample result
                pixel_color = pixel_color + sample_color;
            }

            // Take the average
            pixel_color = pixel_color / float(samples);
            framebuffer[y * width + x] = pixel_color;
        }
    }

    save_image(framebuffer.data(), width, height, "output_cpu.ppm");
    // Output the total execution time
    auto end = std::chrono::high_resolution_clock::now();

    // Output the total execution time
    std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms\n";

    return 0;
}
