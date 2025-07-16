# Isaac Arena

## Overview

Isaac Arena aims to enhance Isaac Lab by providing a scalable environment creation and evaluation framework.
This project simplifies the process of generating diverse simulation environments and evaluating robot learning policies.

## Purpose

- Provide a greater number of environments out of the box.
- Offer a simple evaluation framework for user policies.

## Features (In Progress)

- **Environment Library**: Pre-packaged environments, tasks, scenes, and embodiments.
- **Agentic Scene Remixer**: Automated variation of scenes.
- **Evaluator**: Framework for evaluating user-supplied policies on diverse environments.

## Development Setup

### Docker Environment

The project uses Docker to provide a consistent development environment with all dependencies pre-installed.

#### Running the Docker Container

1. **Run the Docker container**:
   ```bash
   ./docker/run_and_push_docker.sh
   ```

### Pre-commit Hooks

#### Installation

1. **Install pre-commit** (if not already installed):
   ```bash
   pip install pre-commit
   ```

2. **Install the git hooks**:
   ```bash
   pre-commit install
   ```

3. **Manual - All Files**:
   ```bash
   pre-commit run --all-files
   ```
