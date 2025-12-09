# 07. Support & BSC Chatbot

## 1. BSC Chatbot (`bsc_chat`)
MN5 hosts an internal AI chatbot for support and general queries, running on local hardware (data privacy compliant).

### Availability
*   **Nodes**: Available on basic login nodes (`glogin1/2`, `alogin1/2`).
*   **Not Available**: Restricted login nodes (`glogin4`, `alogin4`).

### Usage
1.  **Load the Module**:
    ```bash
    module load bsc
    ```
2.  **Start Chat**:
    ```bash
    bsc_chat
    ```
3.  **Select Model**:
    *   `0: Knowledge-support`: Trained on BSC docs. Best for "How do I use SLURM?" questions.
    *   `1: Llama-70B-Instruct`: General purpose reasoning.

### Interaction
*   Type your prompt.
*   Press **ALT + ENTER** to send.

> [!NOTE]
> Like all LLMs, it can hallucinate. Always verify commands with official docs.

---

## 2. Official Support
If the chatbot or docs don't help:
*   **Email**: support@bsc.es
*   **Ticket**: Include your username (`bscXXYY`) and Job ID if relevant.
