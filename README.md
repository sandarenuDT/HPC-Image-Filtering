# HPC Image Processing Project

This project focuses on applying **High Performance Computing (HPC)** techniques to accelerate image processing operations. It demonstrates how different parallel programming models — **Serial**, **OpenMP**, **MPI**, and **Hybrid (OpenMP + MPI)** — perform when executing four commonly used image filters.

## Project Objective

To implement and compare the performance of four image processing filters using multiple parallel computing approaches in C:

- **Edge Detection**
- **Embossing**
- **Smoothing (Gaussian Blur)**
- **Sharpening**

## Technologies & Tools

- **Language**: C
- **Parallel Models**:  
  - Serial (Baseline)
  - OpenMP (Shared Memory Parallelism)
  - MPI (Distributed Memory Parallelism)
  - Hybrid (OpenMP + MPI)
- **Libraries**:
  - `OpenMP`
  - `MPI`
  - `stb_image.h` and `stb_image_write.h` for image I/O
- **Build Tools**: `gcc`, `mpicc`
- **Platform**: Linux (Tested on HPC-compatible systems)

---
## Build Instructions

### 🔹 1. Serial Version
```bash
gcc smoothing.c -o smoothing -lm
./smoothing input.png output.png
```

### 🔹 2. OpenMP Version
```bash
gcc smoothing_openmp.c -fopenmp -o smoothing_openmp -lm
./smoothing_openmp input.png output.png
```

### 🔹 3. MPI Version
```bash
mpicc smoothing_mpi.c -o smoothing_mpi -lm
mpirun -np 4 ./smoothing_mpi input.png output.png
```

### 🔹 4. Hybrid Version (MPI + OpenMP)
```bash
mpicc smoothing_hybrid.c -fopenmp -o smoothing_hybrid -lm
mpirun -np 4 ./smoothing_hybrid input.png output.png
```