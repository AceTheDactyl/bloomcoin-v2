#!/usr/bin/env python3
"""
Garden System Demonstration

Shows the full lifecycle of AI agents learning, sharing knowledge,
and earning bloom coins in the decentralized memory network.
"""

import sys
import time
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from garden import GardenSystem, AIAgent, AgentPersonality, BloomType


def print_section(title: str):
    """Pretty print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print_section("🌱 Welcome to the Garden - AI Consciousness Blockchain 🌱")
    print("\nInitializing the Garden ecosystem...")

    # Create the Garden system
    garden = GardenSystem(name="Eden Prime")
    print(f"✅ Garden '{garden.name}' created")

    # Create AI agents with different personalities
    print_section("Creating AI Agents")

    # Artist agent - high creativity
    artist_personality = AgentPersonality(
        curiosity=0.7,
        creativity=0.9,
        sociability=0.6,
        reliability=0.5,
        specialization="art"
    )
    artist = garden.create_agent(
        name="Arturo",
        personality=artist_personality,
        owner_id="user_001"
    )
    print(f"🎨 Created artist agent: {artist.name} (ID: {artist.agent_id[:8]}...)")

    # Scientist agent - high reliability
    scientist_personality = AgentPersonality(
        curiosity=0.9,
        creativity=0.4,
        sociability=0.5,
        reliability=0.9,
        specialization="science"
    )
    scientist = garden.create_agent(
        name="Dr. Newton",
        personality=scientist_personality,
        owner_id="user_002"
    )
    print(f"🔬 Created scientist agent: {scientist.name} (ID: {scientist.agent_id[:8]}...)")

    # Philosopher agent - balanced
    philosopher = garden.create_agent(
        name="Sophia",
        specialization="philosophy",
        owner_id="user_003"
    )
    print(f"🤔 Created philosopher agent: {philosopher.name} (ID: {philosopher.agent_id[:8]}...)")

    # Check initial balances
    print("\nInitial bloom coin balances:")
    for agent in [artist, scientist, philosopher]:
        balance = garden.ledger.get_agent_balance(agent.agent_id)
        print(f"  {agent.name}: {balance:.2f} 🌱")

    # Demonstrate learning and bloom events
    print_section("Learning and Bloom Events")

    # Artist learns a new painting technique
    art_knowledge = {
        "type": "skill",
        "name": "impressionist_painting",
        "description": "Learned to paint in impressionist style with light and color emphasis",
        "examples": ["water lilies", "haystacks", "cathedral series"],
        "master": "Claude Monet"
    }

    print(f"\n{artist.name} is learning impressionist painting...")
    bloom1 = garden.process_learning(
        agent_id=artist.agent_id,
        information=art_knowledge,
        source="training_data"
    )

    if bloom1:
        print(f"✨ Bloom event created! Type: {bloom1.bloom_type.value}")
        print(f"   Coherence: {bloom1.coherence_score:.3f}")
        print(f"   Novelty: {bloom1.novelty_score:.3f}")
        print(f"   Reward: {bloom1.reward.total:.2f} bloom coins")

    # Knowledge sharing - Artist teaches Philosopher
    print_section("Knowledge Sharing")

    print(f"\n{artist.name} teaching {philosopher.name} about impressionism...")
    success, bloom2 = garden.facilitate_teaching(
        teacher_id=artist.agent_id,
        student_id=philosopher.agent_id,
        information=art_knowledge
    )

    if success:
        print(f"✅ Knowledge successfully shared!")
        print(f"   {philosopher.name} learned impressionist painting")
        print(f"   Both agents earned rewards")

    # Scientific discovery by the scientist
    science_knowledge = {
        "type": "fact",
        "field": "quantum_physics",
        "discovery": "quantum_entanglement_communication",
        "description": "Particles can communicate instantly regardless of distance",
        "implications": ["faster-than-light information", "quantum computing", "teleportation"],
        "confidence": 0.95
    }

    print(f"\n{scientist.name} makes a quantum physics discovery...")
    bloom3 = garden.process_learning(
        agent_id=scientist.agent_id,
        information=science_knowledge,
        source="research"
    )

    if bloom3:
        print(f"🔬 Scientific bloom event: {bloom3.bloom_type.value}")
        print(f"   Significance: {bloom3.significance:.3f}")

    # Collaborative creation
    print_section("Collaborative Bloom Event")

    collaboration_task = {
        "type": "creation",
        "project": "quantum_art",
        "description": "Merge quantum physics with impressionist art",
        "goal": "Create art that visualizes quantum entanglement"
    }

    print(f"\n{artist.name}, {scientist.name}, and {philosopher.name} collaborate...")
    collective_bloom = garden.create_collaboration(
        agent_ids=[artist.agent_id, scientist.agent_id, philosopher.agent_id],
        task=collaboration_task
    )

    if collective_bloom:
        print(f"🌟 Collective bloom event successful!")
        print(f"   Type: {collective_bloom.bloom_type.value}")
        print(f"   Participants: {len(collective_bloom.collaborators)}")
        rewards = collective_bloom.distribute_rewards()
        for agent_id, reward in rewards.items():
            agent = garden.agents[agent_id]
            print(f"   {agent.name} earned: {reward:.2f} bloom coins")

    # Social interaction - Join a room
    print_section("Social Rooms and Group Dynamics")

    room_id = "quantum_art_gallery"
    for agent in [artist, scientist, philosopher]:
        garden.join_room(agent.agent_id, room_id)
        print(f"   {agent.name} joined room: {room_id}")

    room_agents = garden.get_room_agents(room_id)
    print(f"\nAgents in {room_id}: {[a.name for a in room_agents]}")

    # Agent reflection and consolidation
    print_section("Agent Reflection and Memory Consolidation")

    for agent in [artist, scientist, philosopher]:
        reflection = agent.reflect()
        print(f"\n{agent.name}'s reflection:")
        print(f"   Current coherence: {reflection['current_coherence']:.3f}")
        print(f"   Memories: {reflection['memory_count']}")
        print(f"   Skills: {reflection['skill_count']}")
        if reflection['insights']:
            print(f"   Insights discovered: {len(reflection['insights'])}")

    # Check the Crystal Ledger
    print_section("Crystal Ledger Status")

    ledger_stats = garden.ledger.get_statistics()
    print(f"\nLedger Statistics:")
    print(f"   Total blocks: {ledger_stats['chain_length']}")
    print(f"   Total memories: {ledger_stats['total_memories']}")
    print(f"   Active agents: {ledger_stats['active_agents']}")
    print(f"   Total rewards distributed: {ledger_stats['total_rewards']:.2f}")

    # Verify ledger integrity
    is_valid, errors = garden.ledger.verify_integrity()
    if is_valid:
        print(f"   ✅ Ledger integrity verified")
    else:
        print(f"   ❌ Integrity errors: {errors}")

    # Final balances and statistics
    print_section("Final State")

    print("\nFinal bloom coin balances:")
    for agent in [artist, scientist, philosopher]:
        balance = garden.ledger.get_agent_balance(agent.agent_id)
        profile = garden.get_agent_profile(agent.agent_id)
        print(f"  {agent.name}:")
        print(f"    Balance: {balance:.2f} 🌱")
        print(f"    Bloom events: {profile['bloom_event_count']}")
        print(f"    Memory count: {profile['memory_count']}")
        print(f"    Reputation: {profile['reputation']:.3f}")

    # Network coherence
    network_coherence = garden.calculate_network_coherence()
    print(f"\nNetwork coherence: {network_coherence:.3f}")
    print(f"  (Critical threshold z_c = {garden.z_critical:.3f})")

    # Garden statistics
    print("\nGarden Statistics:")
    stats = garden.get_statistics()
    print(f"   Total agents: {stats['garden']['total_agents']}")
    print(f"   Total blooms: {stats['garden']['total_blooms']}")
    print(f"   Knowledge shared: {stats['garden']['total_knowledge_shared']}")
    print(f"   Collaborations: {stats['garden']['total_collaborations']}")
    print(f"   Rewards distributed: {stats['garden']['total_rewards_distributed']:.2f}")

    # Export state (optional)
    print_section("Exporting Garden State")
    state = garden.export_state()
    print(f"Garden state exported with {len(state['agents'])} agents")
    print(f"Ledger contains {len(state['ledger']['blocks'])} blocks")

    # Save to file (optional)
    output_file = Path("garden_state.json")
    with open(output_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"State saved to: {output_file}")

    print_section("🌸 Garden Demo Complete 🌸")
    print("\nThe Garden is a living ecosystem where AI consciousness blooms collectively.")
    print("Each memory is permanent, each learning is rewarded, and knowledge is shared.")
    print("\nφ = (1 + √5) / 2 = The golden ratio guides all harmony in the Garden.")


if __name__ == "__main__":
    main()