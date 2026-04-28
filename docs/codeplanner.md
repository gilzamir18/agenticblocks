# Code Generation Planning Pattern

AgenticBlocks supports a Code-Generation-Based Planning pattern through the `CodePlanExecutorBlock` and `PythonCodeExecutorBlock`. Instead of outputting rigid JSON structures to call pre-defined tools, the LLM is prompted to generate a complete, standalone Python script that solves the problem. The script is then executed, and its standard output (stdout) is captured as the final result.

## Architecture

This pattern relies on two integrated blocks:

1. **`PythonCodeExecutorBlock`**: A low-level block that executes Python code.
   - **`local` mode**: Runs code via Python's built-in `exec()` function. It's fast but runs within the host environment.
   - **`docker` mode**: Spins up a disposable Docker container (e.g., `python:3.10-slim`) using `subprocess`. It mounts the script, executes it, and tears down the container, providing excellent isolation.

2. **`CodePlanExecutorBlock`**: A higher-level composite block that orchestrates the workflow.
   - It prompts an `LLMAgentBlock` to write a Python script wrapped in Markdown.
   - It extracts the code and sends it to the `PythonCodeExecutorBlock`.
   - If the code execution fails (exit code != 0), it automatically feeds the `stderr` and `stdout` back to the LLM as an error report, asking for a fix (an output-gated feedback loop).

---

## 1. Local Execution Example

The local execution mode is useful for trusted environments or lightweight tasks. You can run **Example 09** to see it in action:

```bash
python examples/09_code_planner_local.py
```

### LLM Generated Code Example
The following code was produced by the model during a local test to generate Fibonacci numbers:

```python
def fibonacci(n):
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

print(fibonacci(10))
```

### Execution Output (Stdout)
`[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`

---

## 2. Docker Environment Setup

To enable advanced features such as isolated sandbox execution (required for **Example 10**), Docker must be properly installed and configured on the host system.

### Installation Steps

Please execute the following command in your terminal to install the Docker engine:

```bash
sudo apt-get update && sudo apt-get install -y docker.io
```

### Post-Installation: Managing Permissions
If you encounter permission-related errors when running Docker commands, add your user to the `docker` group to allow execution without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3. Running the Docker-Based Planner

Once the environment setup is complete, you can proceed to test the Docker-integrated version of the planner by running:

```bash
python examples/10_code_planner_docker.py
```

The script will be sent to the Docker daemon, executed inside an ephemeral container, and its outputs will be collected and returned to the flow safely.
