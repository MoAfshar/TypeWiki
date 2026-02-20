# TypeWiki
Copilot for Typeform's help center common questions

## Getting Started

### Prerequisites

Install [uv](https://github.com/astral-sh/uv), a fast Python package installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.local/bin/env
```

### Installation

1. **Create a virtual environment:**
   ```bash
   make venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install the project with dev dependencies:**
   ```bash
   make install-dev-local
   ```

This will install all dependencies, set up pre-commit hooks, and prepare your development environment.
