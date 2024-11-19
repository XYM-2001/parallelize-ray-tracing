#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <curand_kernel.h>

// Vec3 class for vector operations
struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    __host__ __device__ float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 normalize() const {
        float mag = sqrt(x * x + y * y + z * z);
        return *this / mag;
    }
};

// Sphere structure
struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
    __host__ __device__ bool intersect(const Vec3& origin, const Vec3& dir, float& t) const {
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

// CUDA kernel for rendering
__global__ void render(Vec3* framebuffer, int width, int height, Sphere* spheres, int sphere_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Vec3 pixel_color(0, 0, 0);
    int samples = 4; // 4 samples per pixel

    // Initialize the random number seed (if random sampling is required)
    curandState rand_state;
    curand_init((unsigned long long)idx, 0, 0, &rand_state);

    for (int s = 0; s < samples; ++s) {
        // Random sampling u and v
        float u = (x + curand_uniform(&rand_state)) / width;
        float v = (y + curand_uniform(&rand_state)) / height;

        // Generate the ray
        Vec3 ray_origin(0, 0, 0);
        Vec3 ray_dir = Vec3(u - 0.5f, v - 0.5f, -1).normalize();

        // Ray-sphere intersection check
        float t_min = 1e20f;
        Vec3 sample_color(0, 0, 0);
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

    // Write the pixel color to the framebuffer
    framebuffer[idx] = pixel_color;
}

// Host function
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

    Vec3* framebuffer;
    cudaMallocManaged(&framebuffer, num_pixels * sizeof(Vec3));

    Sphere spheres[] = {
        { Vec3(0, 0, -3), 1, Vec3(1, 0, 0) },
        { Vec3(2, 0, -4), 1, Vec3(0, 1, 0) },
        { Vec3(-2, 0, -4), 1, Vec3(0, 0, 1) }
    };
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(spheres));
    cudaMemcpy(d_spheres, spheres, sizeof(spheres), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    render<<<blocks, threads>>>(framebuffer, width, height, d_spheres, 3);

    cudaDeviceSynchronize();

    save_image(framebuffer, width, height, "output.ppm");

    cudaFree(framebuffer);
    cudaFree(d_spheres);

    // Calculate the execution time
    auto end = std::chrono::high_resolution_clock::now();

    // Output the execution time
    std::cout << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << " ms\n";
            
    return 0;
}
