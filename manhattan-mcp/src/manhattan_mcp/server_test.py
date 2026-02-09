import asyncio
import json
import sys
import os

# Ensure we can import from the current directory and find gitmem
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# Also need to find gitmem which might be relative
gitmem_path = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(gitmem_path)

try:
    from server import (
        create_memory, process_raw_dialogues, add_memory_direct, search_memory,
        get_context_answer, update_memory_entry, delete_memory_entries, chat_with_agent,
        create_agent, list_agents, get_agent, update_agent, disable_agent, enable_agent, delete_agent,
        agent_stats, list_memories, bulk_add_memory, export_memories, import_memories,
        memory_summary, api_usage, auto_remember, should_remember, get_memory_hints,
        conversation_checkpoint, check_session_status, session_start, session_end,
        pull_context, push_memories, get_startup_instructions, request_agent_id,
        pre_response_check, what_do_i_know, mystery_peek
    )
except ImportError as e:
    print(f"Import Error: {e}")
    # Try importing as package if running from root
    from manhattan_mcp.server import (
        create_memory, process_raw_dialogues, add_memory_direct, search_memory,
        get_context_answer, update_memory_entry, delete_memory_entries, chat_with_agent,
        create_agent, list_agents, get_agent, update_agent, disable_agent, enable_agent, delete_agent,
        agent_stats, list_memories, bulk_add_memory, export_memories, import_memories,
        memory_summary, api_usage, auto_remember, should_remember, get_memory_hints,
        conversation_checkpoint, check_session_status, session_start, session_end,
        pull_context, push_memories, get_startup_instructions, request_agent_id,
        pre_response_check, what_do_i_know, mystery_peek
    )

async def test_all_tools():
    agent_id = "test-server-agent-001"
    print(f"=== Starting Server Test with Agent ID: {agent_id} ===\n")

    # Helper to print formatted JSON
    def print_res(name, res):
        try:
            parsed = json.loads(res)
            print(f"✅ {name}:\n{json.dumps(parsed, indent=2)}\n")
        except:
            print(f"✅ {name}: {res}\n")

    # 1. Session & Agent Management
    print("--- 1. Session & Agent Management ---")
    
    res = await check_session_status()
    print_res("check_session_status", res)

    # Create agent
    res = await create_agent("Test Server Agent", agent_id, description="Agent for testing server.py")
    print_res("create_agent", res)
    
    # Start session
    res = await session_start(agent_id, auto_pull_context=True)
    print_res("session_start", res)

    res = await get_startup_instructions()
    print_res("get_startup_instructions", res)

    res = await request_agent_id()
    print_res("request_agent_id", res)

    # 2. Memory CRUD
    print("--- 2. Memory CRUD ---")
    
    # Create/Init memory (clear previous)
    res = await create_memory(agent_id, clear_db=True)
    print_res("create_memory (clear=True)", res)

    # Add memory
    memories = [
        {
            "lossless_restatement": "User enjoys testing software servers.",
            "keywords": ["testing", "server", "software"],
            "topic": "preferences"
        },
        {
            "lossless_restatement": "User is currently working on manhattan-mcp integration.",
            "keywords": ["manhattan-mcp", "integration", "work"],
            "topic": "project"
        }
    ]
    res = await add_memory_direct(agent_id, memories)
    print_res("add_memory_direct", res)

    # Bulk add
    res = await bulk_add_memory(agent_id, [
        {"lossless_restatement": "User prefers concise logs.", "keywords": ["logs", "preferences"]},
        {"lossless_restatement": "User uses VS Code.", "keywords": ["VS Code", "tools"]}
    ])
    print_res("bulk_add_memory", res)

    # Auto remember
    res = await auto_remember(agent_id, "My name is ServerTester and I am running a script.")
    print_res("auto_remember", res)

    # Process raw dialogues
    dialogues = [{"content": "I need to finish the API test by 5pm today."}]
    res = await process_raw_dialogues(agent_id, dialogues)
    print_res("process_raw_dialogues", res)

    # Should remember
    res = await should_remember("I like pizza.")
    print_res("should_remember", res)

    # Chat with agent (auto-extract)
    res = await chat_with_agent(agent_id, "Just checking if you are listening.")
    print_res("chat_with_agent", res)

    # 3. Retrieval & Search
    print("--- 3. Retrieval & Search ---")

    res = await search_memory(agent_id, "testing")
    print_res("search_memory ('testing')", res)

    res = await get_context_answer(agent_id, "What is the user working on?")
    print_res("get_context_answer", res)

    res = await memory_summary(agent_id)
    print_res("memory_summary", res)

    res = await get_memory_hints(agent_id)
    print_res("get_memory_hints", res)

    res = await what_do_i_know(agent_id)
    print_res("what_do_i_know", res)

    res = await mystery_peek("Project deadline", agent_id)
    print_res("mystery_peek", res)
    
    res = await pre_response_check("Who are you?", "Identity check")
    print_res("pre_response_check", res)

    res = await pull_context(agent_id)
    print_res("pull_context", res)

    # 4. Management & Stats
    print("--- 4. Management & Stats ---")

    res = await list_agents()
    print_res("list_agents", res)

    res = await get_agent(agent_id)
    print_res("get_agent", res)

    res = await agent_stats(agent_id)
    print_res("agent_stats", res)

    res = await list_memories(agent_id, limit=5)
    print_res("list_memories", res)
    
    # Get an ID to update/delete
    list_res = json.loads(res)
    if list_res.get("memories"):
        mem_id = list_res["memories"][0]["id"]
        
        # Update
        res = await update_memory_entry(agent_id, mem_id, {"lossless_restatement": "Updated: User REALLY enjoys testing software servers."})
        print_res("update_memory_entry", res)
        
        # Delete
        res = await delete_memory_entries(agent_id, [mem_id])
        print_res("delete_memory_entries", res)

    # Checkpoint
    res = await conversation_checkpoint(agent_id, "Server Test Checkpoint", ["Tested CRUD", "Tested Search", "All good"])
    print_res("conversation_checkpoint", res)

    # Export/Import
    res = await export_memories(agent_id)
    # print_res("export_memories", res) # verbose
    export_data = json.loads(res)
    print(f"✅ export_memories: Retrieved {len(export_data.get('memories', []))} memories from export.\n")
    
    if export_data.get("memories"):
        res = await import_memories(agent_id, export_data, merge_mode="append")
        print_res("import_memories", res)

    # Push memories (sync)
    res = await push_memories(agent_id)
    print_res("push_memories", res)

    # Usage
    res = await api_usage()
    print_res("api_usage", res)

    # Agent Lifecycle (simulated)
    res = await update_agent(agent_id, {"description": "Updated Description"})
    print_res("update_agent", res)
    
    res = await disable_agent(agent_id)
    print_res("disable_agent", res)
    
    res = await enable_agent(agent_id)
    print_res("enable_agent", res)

    # End Session
    res = await session_end(agent_id, "Server Test Complete", ["All functions verified"])
    print_res("session_end", res)

    # Cleanup (Optional)
    # res = await delete_agent(agent_id)
    # print_res("delete_agent", res)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_all_tools())
