1. ssh cuda2
2. module load cuda-12.4
3. nvcc ray_tracing.cu -o ray_tracing
4. nvcc ray_tracing_cpu.cu -o ray_tracing_cpu
5. ./ray_tracing
6. ./ray_tracing_cpu