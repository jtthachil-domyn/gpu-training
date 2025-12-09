# 00. New Essential Changes (MN5 vs older)

If you are coming from MareNostrum 4 (MN4) or other clusters, pay attention.

## 1. User & Account Management
*   **Username Change**: `bscXXYYY` -> `bsc0XXYYY` (for some staff).
*   **Groups**:
    *   **Primary Group**: Institutional (no resources).
    *   **Secondary Groups**: Projects (where quota/jobs live).
*   **Action**: You MUST use `newgrp` (Linux) or `--account` (Slurm) to target the correct project group.

## 2. SLURM Submission (Mandatory fields)
You **must** specify both:
1.  `--account=<secondary_group>`
2.  `--qos=<queue_name>`

## 3. Storage Hierarchy
*   **Home**: `/gpfs/home` (One per user).
*   **Projects**: `/gpfs/projects` (One per project group).
*   **Scratch**: `/gpfs/scratch` (One per project group).

## 4. Performance Critical: `srun` & CPU Binding
**Change**: `srun` no longer inherits `--cpus-per-task` from `sbatch` automatically in the newer Slurm version.

**Fix**: You must be explicit or export the variable.

**Bad (Likely to fail or run slow):**
```bash
#SBATCH --cpus-per-task=20
srun ./my_binary
```

**Good:**
```bash
#SBATCH --cpus-per-task=20
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
srun params...
```

**Reason**: Without this, thread affinity (pinning) breaks, and threads may overlap on the same core.

## 5. MPI & NVIDIA SDK
*   If using **NVIDIA HPC SDK**, use `mpirun` instead of `srun`.
*   Set `export SLURM_CPU_BIND=none` when using `mpirun`.
