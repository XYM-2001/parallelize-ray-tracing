#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

// Vec3 and Sphere remain the same as in your CUDA code
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

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    bool intersect(const Vec3& origin, const Vec3& dir, float& t) const {
        Vec3 oc = origin - center;
        float a = dir.dot(dir);
        float b = 2.0f * oc.dot(dir);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return false;
        t = (-b - sqrt(discriminant)) / (2.0f * a);
        return t > 0;
    }
};

// Save image function remains the same
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
    int width = 7680;
    int height = 4320;
    int num_pixels = width * height;\

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Vec3> framebuffer(num_pixels); // 使用 std::vector 存储像素

    Sphere spheres[] = {
        { Vec3(0, 0, -3), 1, Vec3(1, 0, 0) },
        { Vec3(2, 0, -4), 1, Vec3(0, 1, 0) },
        { Vec3(-2, 0, -4), 1, Vec3(0, 0, 1) }
    };
    int sphere_count = 3;

    // 遍历每个像素
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float u = (x + 0.5f) / width;
            float v = (y + 0.5f) / height;
            Vec3 ray_origin(0, 0, 0);
            Vec3 ray_dir = Vec3(u - 0.5f, v - 0.5f, -1).normalize();

            Vec3 pixel_color(0, 0, 0);
            float t_min = 1e20f;

            // 遍历每个球体
            for (int i = 0; i < sphere_count; ++i) {
                float t;
                if (spheres[i].intersect(ray_origin, ray_dir, t) && t < t_min) {
                    t_min = t;
                    pixel_color = spheres[i].color;
                }
            }

            framebuffer[idx] = pixel_color;
        }
    }

    save_image(framebuffer.data(), width, height, "output_cpu.ppm");
    // 停止计时
    auto end = std::chrono::high_resolution_clock::now();

    // 输出时间
    std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms\n";

    return 0;
}
