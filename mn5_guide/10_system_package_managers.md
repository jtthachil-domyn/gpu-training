# 10. System Package Managers (Spack, EasyBuild, EESSI)

Beyond standard modules and Python/Conda, MN5 provides access to three powerful package management systems. These are useful if you need to compile scientific software from source with specific optimizations or dependencies not found in the default environment.

## 1. EESSI (European Environment for Scientific Software Installations)
**Best for**: Instant access to a huge library of scientific software optimized for x86_64, ARM, and GPUs, streamed over the network.

*   **How it works**: Uses CVMFS to mount a read-only repository at `/cvmfs/software.eessi.io`.
*   **Usage**:
    1.  **Load the Module**:
        ```bash
        module load eessi
        ```
    2.  **Search & Load Software**:
        ```bash
        module avail
        module load GROMACS/2024.1-foss-2023b
        ```
*   **Partition Support**:
    *   **GPP**: Loads CPU-optimized modules `.../sapphirerapids/modules/all`.
    *   **ACC**: Loads CPU modules + GPU-optimized modules `.../sapphirerapids/accel/nvidia/cc90/modules/all`.

## 2. EasyBuild
**Best for**: Consistency. The primary tool used by BSC admins to install the system modules.

*   **Usage**:
    1.  **Load the Module**:
        ```bash
        module load EB/apps
        ```
    2.  **Search & Load**:
        ```bash
        module avail
        module load <name>/<version>  # ALWAYS specify version!
        ```
*   **Locations**:
    *   **GPP**: `/apps/GPP/EASYBUILD/modules/all`
    *   **ACC**: `/apps/ACC/EASYBUILD/modules/all`

## 3. Spack
**Best for**: "Do It Yourself" custom builds. If you need a specific version of a library with a non-standard compiler flag.

*   **System Spack**:
    ```bash
    module load spack
    spack find        # List installed packages
    spack load <name> # Load a package
    ```

*   **User Spack (Advanced)**:
    You can install your own Spack instance in your home directory and "chain" it to the system Spack to reuse pre-built packages.
    1.  Install Spack in `$HOME`.
    2.  Configure `$HOME/spack/etc/spack/defaults/upstreams.yaml`:
        ```yaml
        upstreams:
          system_spack:
            install_tree: /apps/GPP/SPACK/0.21.2/opt/spack
        ```
    3.  Install your custom packages: `spack install my-package`
