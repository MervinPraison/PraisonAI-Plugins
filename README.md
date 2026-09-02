# PraisonAI Plugins

PraisonAI Plugins is a foundational repository demonstrating how to extend the standard capabilities of the PraisonAI framework using the official **Plugin Ecosystem**.

Every design decision here revolves around our **Agent-Centric Philosophy**. You simply write standard python functions and register them. PraisonAI's architecture handles dynamically searching and integrating your plugins at startup without complex configuration files strings.

## Features

This repository includes examples and templates covering all **6 Plugin Types** categorized natively by the PraisonAI SDK:

1. **Hooks**: Inject code into agent interactions like tracking API calls, displaying terminal logs, etc. (`simple_logger`, `custom_tracer`)
2. **Tools**: Bundle functional python functions as tools that an agent can call automatically. (`basic_tools`)
3. **Guardrails**: Validate outputs, ensure safety, intercept Prompt Injections, prevent PII escapes. (`pii_guardrail`)
4. **Policies**: Set runtime execution contexts—for example blocking dangerous tool execution unconditionally. (`strict_policy`)
5. **Skills**: Bundle higher-level instructions and techniques to transform how the agent "thinks". (`researcher_skill`)
6. **Integrations**: Effortless connectors to 3rd party APIs, internal databases, or services like Slack, Jira. (`slack_integration`)

---

## 🚀 Quickstart for Non-Developers

### Install

Getting started is designed to be frictionless for both beginners and experts:

```bash
# 1. Clone this package (the plugin repository)
git clone https://github.com/mervinpraison/praisonai-plugins.git
cd praisonai-plugins

# 2. Install the plugins to your global python environment
pip install -e .
```

That's it. **PraisonAI automatically knows you installed these plugins.** No messy config required.

### Testing Agent Flow

By simply checking if the plugins are available, your standard PraisonAI `Agent` acts normally, but now runs with the enhanced logic in your Plugins. For example, your `test_plugins_discovery.py` handles:

```python
from praisonaiagents import Agent

agent = Agent(name="Tester", instructions="Just repeat what I say.")
# This automatically executes all logic defined in plugins: Guardrails, Logging, etc
agent.start("Hello world!")
```

---

## 🛠️ Developer Guide: Building Custom Plugins

The PraisonAI SDK natively implements a powerful **Protocol-Driven plugin system**. 

A Protocol refers to a strict interface specification defining *WHAT* functionality exists without touching *HOW* PraisonAI works internally. Your plugins implement hooks across an Agent's lifecycle without modifying core SDK files!

### 1. Write the Plugin

Inside the `src/praisonai_plugins` folder, you can create a standard python class mimicking this example Guardrail plugin (`pii_guardrail.py`):

```python
from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook

class PIIGuardrailPlugin(Plugin):
    """A protocol-driven plugin evaluating guardrails."""
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="pii_guardrail",
            version="1.0.0",
            description="Blocks PII in responses.",
            author="PraisonAI",
            hooks=[PluginHook.AFTER_LLM] # Define when this plugin acts!
        )
        
    def after_llm(self, response: str, usage: dict) -> str:
        # Example Logic protecting SSN
        if "social security" in response.lower():
            return "[REDACTED BY GUARDRAIL]"
        return response
```

### 2. Export it via `pyproject.toml`

PraisonAI scans for `[project.entry-points."praisonai.plugins"]` to load classes automatically! In the `pyproject.toml` root file, export it:

```toml
[project.entry-points."praisonai.plugins"]
pii_guardrail = "praisonai_plugins.guardrails.pii_guardrail:PIIGuardrailPlugin"
```

### 3. Reinstall 

```bash
pip install -e .
```

Now, every time *any* agent inside your Python environment receives a response containing "social security", this plugin intercepts it, evaluates it, and scrubs the PII.

---

## Sandbox Backends (`praisonai.sandbox`)

Optional sandbox backends (e.g. **Capsule** for WebAssembly isolation) live in this repo and register via the **`praisonai.sandbox`** entry-point group — separate from lifecycle hooks (`praisonai.plugins`).

```bash
pip install praisonai-plugins[capsule]
```

```python
from praisonaiagents.sandbox import SandboxManager, SandboxConfig

manager = SandboxManager(SandboxConfig.capsule())
result = manager.run_code("print('hello')", language="python")
```

| Entry point group | Purpose | Example |
|-------------------|---------|---------|
| `praisonai.plugins` | Lifecycle hooks, guardrails, policies | `simple_logger`, `pii_guardrail` |
| `praisonai.sandbox` | Optional sandbox backends | `capsule` |

## AgentFuse Tool Guardrail

The optional AgentFuse middleware evaluates a tool-call policy through
PraisonAI's public `wrap_tool_call` boundary before the protected handler starts.

```bash
pip install -e ".[agentfuse]"
```

```python
from dhms_agentfuse import RuntimeGuard
from praisonaiagents import Agent
from praisonai_plugins.guardrails.agentfuse import AgentFuseToolMiddleware

guard = RuntimeGuard(deny_tools={"protected_write"})
agent = Agent(
    name="guarded-agent",
    instructions="Use tools when needed.",
    hooks=[AgentFuseToolMiddleware(guard)],
)
```

An allowed call continues through PraisonAI's existing handler chain. A blocked
call, missing tool-call identity, or unexpected guard-evaluation exception returns
a host-native `ToolResponse` with `outcome=not_executed`; it does not claim that a
handler failed after execution began.

---

## The PraisonAI Protocol Philosophy

Built with developers in mind, PraisonAI's architectural design features:
- **Zero Overhead**: The underlying Core SDK limits itself to hooks and standard Dataclass instances. Lazy-imports mean plugin hooks effectively run with near-zero performance cost.
- **Safe Multi-Agent Design**: Plugins don't hold shared state. If you orchestrate `AgentFlow()` loops, these plugins safely execute globally without thread-locking context bugs.
- **Protocol Extensibility**: You're never boxed in. Whether it's the `MemoryProtocol`, `ToolProtocol`, or native events (`before_agent`, `after_llm`), the Plugin abstraction matches any scenario you face in building robust GenAI apps.
