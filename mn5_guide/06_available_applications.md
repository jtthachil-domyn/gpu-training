# 06. Available Applications & Benchmarks (Detailed)

MN5 hosts optimized builds for major scientific domains.

## 1. Domain Specific Applications
Checking availability: `module avail <name>`

### Chemistry/Materials
*   **GROMACS**:
    *   **Module**: `module load GROMACS/2024.1-acc-cuda-12` (Example).
    *   **Tip**: Always use the `-acc-cuda` versions on ACC partition for 50x speedup over CPU versions.
    *   **Command**: `gmx_mpi mdrun ...`
*   **VASP**:
    *   **Module**: `module load VASP`
    *   **License**: Restricted. You must be in the `vasp` unix group.
    *   **Tip**: VASP requires `ulimit -s unlimited` in your script stack size.
*   **CP2K**:
    *   Hybrid MPI/OpenMP is critical here. Set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.

> [!NOTE]
> For detailed benchmarks and build references, see the official application pages: [GPP Apps](https://www.bsc.es/supportkc/docs/MareNostrum5/Marenostrum5-Applications/GPP), [ACC Apps](https://www.bsc.es/supportkc/docs/MareNostrum5/Marenostrum5-Applications/ACC/).

## 2. Deep Learning / AI
BSC provides pre-built containers and modules, but most users bring their own environment.

### Recommended AI Stack
1.  **Base**: `module load python/3.11` + `module load cuda/12.1`
2.  **Manager**: `venv` (preferred) or `conda`.
3.  **Flash Attention**: Installing `flash-attn` can be tricky. Ensure `nvcc --version` matches your PyTorch CUDA version exactly.

## 3. Benchmarks
*   **HPL (High Performance Linpack)**: `module load hpl`. Used for stress testing.
*   **HPCG**: Memory-bound benchmark.
