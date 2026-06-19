# Semantic Loop Detector Plugin - Implementation Summary

## What I Learned About PraisonAI's Plugin Architecture
The plugin system is deeply integrated with the `praisonaiagents` core via Python `entry_points` in the `[project.entry-points."praisonai.plugins"]` namespace. 
Plugins inherit from the `praisonaiagents.plugins.plugin.Plugin` base class and must expose an `@property def info(self) -> PluginInfo:` descriptor containing metadata and a list of requested `PluginHook` events. 

At runtime, the `PluginManager` discovers and initializes these plugins dynamically, routing agent execution steps (like `AFTER_LLM` or `BEFORE_TOOL`) to the registered hooks.

## How My Plugin Integrates
The `SemanticLoopDetectorPlugin` hooks strictly into the `PluginHook.AFTER_LLM` lifecycle event. It inspects the `response` string returned by the model before it reaches the core agent flow. 
By maintaining an internal `deque` history of the last 5 outputs, it tokenizes the text, generates bigrams, and runs a zero-dependency Jaccard similarity comparison.

When the Jaccard similarity between the current output and any previous output exceeds the 0.85 threshold, the plugin dynamically intercepts the output and appends a `[SYSTEM INTERVENTION]` directive. This securely forces the agent to break its reasoning loop without crashing the process.

## Assumptions Made
1. **History Scope**: I assumed that preserving state locally on the plugin instance (`self.history`) is safe because the plugin instance persists across the execution lifecycle of the agent running within the same process.
2. **Intervention Method**: I assumed that injecting an inline `[SYSTEM INTERVENTION]` directly into the `response` string returned from `AFTER_LLM` is the preferred way to natively steer the agent, as PraisonAI passes this returned text back into the LLM context.
3. **No External Libraries**: I adhered strictly to the "zero dependencies" rule. As a result, the "tokenization" process splits by standard spaces and strips punctuation using `re`, avoiding the need for heavy libraries like `tiktoken` or `nltk`.

## Questions/Concerns for the Maintainers
1. **Multi-Agent State Isolation**: Does `PluginManager` instantiate a unique plugin object *per agent* or *per process*? If it's a singleton per process, the `self.history` deque might incorrectly blend the outputs of multiple concurrent agents. I may need to key the `self.history` dictionary by `agent_id` or `session_id` if that is the case.
2. **Hook Execution Order**: In `AFTER_LLM`, if multiple plugins (like `CustomTracerPlugin` or `PIIGuardrailPlugin`) are active, what dictates the execution order? I want to ensure the `[SYSTEM INTERVENTION]` string is appended *before* metrics tracking evaluates the final payload.
