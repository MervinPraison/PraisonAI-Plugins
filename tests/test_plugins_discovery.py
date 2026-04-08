from praisonaiagents.plugins.manager import get_plugin_manager
from praisonaiagents import Agent

def main():
    manager = get_plugin_manager()
    print("Discovering entry points...")
    loaded = manager.discover_entry_points()
    print(f"Loaded {loaded} plugins from entry points.")
    
    # Just to confirm the hooks are wired up
    print("Registered plugins:")
    for plugin_id, plugin in manager._plugins.items():
        print(f" - {plugin.info.name} (version {plugin.info.version})")
        
    print("\nExecuting agent test:")
    agent = Agent(name="Tester", instructions="Just repeat what I say.")
    # This should trigger SimpleLogger and CustomTracer hooks
    agent.start("Hello world!")
    
if __name__ == "__main__":
    main()
