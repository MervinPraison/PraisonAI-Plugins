# PraisonAI Plugins

Protocol-driven plugins for PraisonAI Agents. This repository contains officially supported plugins that hook into the core agent lifecycle events.

## Installation

```bash
pip install praisonai-plugins
```

If you are developing plugins locally:
```bash
git clone https://github.com/MervinPraison/PraisonAI-Plugins.git
cd PraisonAI-Plugins
pip install -e .
```

## How It Works

PraisonAI's core SDK introduces a `discover_entry_points()` mechanism in its `PluginManager`. 
When you install `praisonai-plugins`, the entry points declared in `pyproject.toml` are registered in your Python environment under the `praisonai.plugins` group.

To load them in your agent:

```python
from praisonaiagents.plugins.manager import get_plugin_manager

manager = get_plugin_manager()
manager.discover_entry_points()

# Your agent will now use the installed plugins!
from praisonaiagents import Agent
agent = Agent(name="Assistant", instructions="Be helpful.")
agent.start("Hello!")
```

## Adding New Plugins

1. Create your plugin class extending `praisonaiagents.plugins.plugin.Plugin`.
2. Implement your desired hooks (`before_agent`, `after_tool`, etc.).
3. Add an entry to the `[project.entry-points."praisonai.plugins"]` section in `pyproject.toml`.
4. Run `pip install -e .` to register your new entry point during testing.
