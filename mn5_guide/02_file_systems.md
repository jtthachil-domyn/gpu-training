# 02. File Systems & Storage Usage

## Storage Hierarchy Summary

| Filesystem | Path | Quota (Soft/Hard) | Speed | Backup | Cleaning Policy | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Home** | `/gpfs/home/bscXX/bscXXYY` | 80GB / 84GB | Low (NFS-like) | **Yes** | Never | Configs, Scripts, Keys, `.bashrc`. |
| **Projects** | `/gpfs/projects/bscXX` | Large (TB-PB) | Medium | **Yes** | Never | Code, Final Models, Read-Only Datasets. |
| **Scratch** | `/gpfs/scratch/bscXX` | Massive (PB) | **High (NVMe)** | **NO** | **2 Weeks** | **Active Training**, Checkpoints, Logs. |


Understanding the storage hierarchy is critical to avoid data loss and quota issues.
management.

## 1. `/gpfs/home`
*   **Path**: `/gpfs/home/bscXX/bscXXYY`
*   **Use Case**: Source code (`.py`, `.sh`), config files, virtual envs.
*   **Quota**: Very small (tens of GBs). **Do not store datasets here.**
*   **Backup**: Yes.

## 2. `/gpfs/projects`
*   **Path**: `/gpfs/projects/bscXX/domyn-guard`
*   **Use Case**: Large shared datasets, stable model checkpoints, finalized results.
*   **Quota**: Large (Terabytes). Shared by the whole team.
*   **Backup**: Often Daily/Weekly (check specifics).

## 3. `/gpfs/scratch` (Critical)
*   **Path**: `/gpfs/scratch/bscXX/bscXXYY` (or shared group scratch)
*   **Use Case**: **Active training I/O**.
    *   Dataset cache (HuggingFace `.cache`)
    *   Training logs (Tensorboard/WandB)
    *   Emergency checkpoints
*   **Performance**: Optimized for high throughput (NVMe enabled on calculation nodes).
*   **Retention**: **NO BACKUP**. Files older than X days (usually 30) are **AUTOMATICALLY DELETED**.

## Recommendation for Training
1.  **Code**: Keep in `$HOME`.
2.  **Data**: Copy dataset from `/gpfs/projects` to `/gpfs/scratch` before training if possible, or read directly from `/gpfs/projects` if read-only performance is sufficient.
3.  **Checkpoints**: Save to `/gpfs/scratch` during run. Copy **only the best** checkpoint to `/gpfs/projects` after training finishes.
