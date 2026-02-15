# Mirai - AI Collaborator & Co-creation Platform

> **Mirai**: Empowering AI Collaborators to work as true partners in co-creation.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://astral.sh/uv/) (Package Manager)

### Environment Setup

1. Clone the repository.
2. Run the environment setup script:
   ```bash
   ./setup_env.sh
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Project Structure

- `mirai/`: Core Python application (FastAPI).
- `watchdog/`: Layer 0 Rust watchdog service.
- `db/`: Database migrations and configurations.
- `docker-compose.yml`: Infrastructure definitions.

## Development

> **All commands must be run via `uv run`** to ensure the correct virtual
> environment and dependencies are used. The system Python will not have
> packages like `structlog` installed.

```bash
# Run the application
uv run main.py

# Run tests
uv run pytest tests/ -q

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## License

This project is licensed under the terms of the LICENSE file.
