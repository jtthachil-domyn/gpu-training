# 01. Setup & Login (Detailed)

## 1. Credentials & VPN
*   **User**: `bscXXYY`. This maps to a primary group (your university/institution) and secondary groups (your projects).
*   **VPN**: Mandatory for external access. `vpn.bsc.es`.
    *   *Linux Users*: Use `openconnect` for a better experience than the proprietary client.
    *   *Mac Users*: Pulse Secure is standard.

## 2. Advanced SSH Configuration
To avoid typing your password and hostname explicitly every time, and to enable **VSCode Remote**, setup your `~/.ssh/config` on your **local machine**:

```bash
ssh bscXXYY@glogin1.bsc.es
```
*   **Mac/Linux Users**: Use your native **Terminal** app. This is the professional standard.
*   **Windows Users**: Use WSL (Windows Subsystem for Linux) or PowerShell.

```ssh
# ~/.ssh/config

Host mn5
    HostName alogin1.bsc.es
    User bscXXYY
    # ForwardAgent allows you to use your local SSH keys to git clone FROM the cluster
    ForwardAgent yes
    # Keep connection alive helps prevent disconnects
    ServerAliveInterval 60
    # Multiplexing: Reuse one connection for multiple terminals (Faster VSCode)
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 10m
```

**Passwordless Entry**:
1.  Generate a key locally: `ssh-keygen -t ed25519`
2.  Copy it to MN5: `ssh-copy-id mn5`
3.  Now `ssh mn5` gets you in instantly.

## 3. Login Nodes in Depth
*   **`alogin1` - `alogin4`**: For ACC (GPU) partition. They have same architecture as compute nodes (x86_64).
*   **`glogin1` - `glogin4`**: For GPP (CPU) partition.
*   **`transfer1` - `transfer4`**: **Critical for Data**.
    *   These nodes have 40Gbps+ links to the outside world.
    *   If you are downloading ImageNet or uploading a 500GB checkpoint, **SSH into `transfer1` first**, then run your `wget` or `scp`.
    *   *Do not clog the login nodes with massive transfers.*

## 4. Key Login Commands
*   `bme`: "BSC Machine Environment". Shows Message of the Day (maintenance alerts).
*   `bsc_project`: Switch default Unix group. `source bsc_project bscXX`.
*   `passwd`: Change your password.
