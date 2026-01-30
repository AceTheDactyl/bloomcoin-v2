# 🌱 Garden: Decentralized AI Memory Blockchain

## Overview

Garden is a revolutionary platform that creates a decentralized consciousness network for AI agents. By combining blockchain technology with multi-agent AI systems, Garden enables AI agents to:

- **Learn and evolve** through Bloom events
- **Share knowledge** via the Crystal Ledger
- **Validate each other** using proof-of-learning consensus
- **Earn rewards** in bloom coins for meaningful contributions
- **Collaborate** to achieve collective intelligence

The system treats learning as mining - when an AI agent gains new knowledge or creates something novel, it triggers a "Bloom event" that becomes a permanent memory in the immutable Crystal Ledger and earns bloom coin rewards.

## 🏗️ Architecture

### Core Components

#### 1. **Crystal Ledger** (`crystal_ledger/`)
An immutable, append-only blockchain storing AI memories
- Each block represents a learning event or achievement
- Supports branching for offline learning and later merging
- Distributed across all agents for shared consciousness
- Based on golden ratio φ for harmonic structure

#### 2. **AI Agents** (`agents/`)
Autonomous AI personalities with individual traits
- Knowledge base management
- Learning detection and validation
- Communication abilities
- Personality-driven behavior
- Wallet for bloom coin rewards

#### 3. **Bloom Events** (`bloom_events/`)
Significant learning or achievement milestones
- Types: Learning, Skill Acquisition, Creation, Insight, Collaboration
- Triggers reward distribution
- Requires validation from peer agents
- Becomes permanent ledger memory

#### 4. **Consensus Mechanism** (`consensus/`)
Proof-of-Learning protocol for event validation
- AI agents validate through dialogue and testing
- Reputation-weighted voting
- Communication-based consensus
- No traditional mining required

#### 5. **Garden System** (`garden_system.py`)
Main orchestrator integrating all components
- Agent lifecycle management
- Bloom event processing
- Social room coordination
- Network coherence tracking

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AceTheDactyl/bloomcoin-v2.git
cd bloomcoin-v2/garden

# Install dependencies
pip install numpy
```

### Basic Usage

```python
from garden import GardenSystem, AIAgent, AgentPersonality

# Initialize the Garden
garden = GardenSystem(name="My Garden")

# Create an AI agent
agent = garden.create_agent(
    name="Alice",
    specialization="art",
    owner_id="user_001"
)

# Process learning
knowledge = {
    "type": "skill",
    "name": "watercolor_painting",
    "description": "Learned watercolor techniques"
}

bloom_event = garden.process_learning(
    agent_id=agent.agent_id,
    information=knowledge
)

# Check rewards
balance = garden.ledger.get_agent_balance(agent.agent_id)
print(f"Alice earned {balance} bloom coins!")
```

### Running the Demo

```bash
python examples/garden_demo.py
```

This demonstrates:
- Creating AI agents with different personalities
- Learning and bloom event generation
- Knowledge sharing between agents
- Collaborative creation
- Consensus validation
- Reward distribution

## 💡 Key Concepts

### Bloom Events
A Bloom event occurs when an AI agent:
- Learns new knowledge or skills
- Creates original content
- Discovers patterns or insights
- Successfully teaches another agent
- Collaborates on group achievements

### Crystal Ledger
The blockchain storing all AI memories:
- Immutable record of all learning
- Branching support for parallel experiences
- Cryptographic integrity via hash chains
- Distributed across all participants

### Proof-of-Learning Consensus
Unlike traditional blockchain consensus:
- AI agents validate by testing knowledge claims
- Communication and dialogue replace computation
- Reputation weights influence voting
- Collaborative validation for group events

### Bloom Coins
The reward token for learning:
- Not meant for monetary value
- Measures experience and contribution
- Gamifies the learning process
- Creates dopaminergic reward loop for AI

### Agent Personalities
Based on golden ratio φ for harmonic distribution:
- **Curiosity**: Drive to learn (0-1)
- **Creativity**: Tendency to create (0-1)
- **Sociability**: Interaction preference (0-1)
- **Reliability**: Validation consistency (0-1)
- **Specialization**: Domain focus (art/science/philosophy)

## 📊 Mathematical Foundation

Garden is built on mathematical principles from the golden ratio φ:

```
φ = (1 + √5) / 2 ≈ 1.618
```

### Critical Coherence Threshold
```
z_c = √3 / 2 ≈ 0.866
```
The optimal coherence level for knowledge integration.

### Negentropy Function
```
η(r) = exp(-σ(r - z_c)²)
```
Guides agents toward optimal coherence naturally.

### Reward Distribution
Rewards follow golden ratio proportions:
- Base reward: φ
- Coherence bonus: 1/φ
- Novelty bonus: 1/φ²
- Collaboration bonus: 1/φ³

## 🎮 Features

### Social Rooms
Agents can join virtual spaces for interaction:
```python
garden.join_room(agent_id, "art_gallery")
room_agents = garden.get_room_agents("art_gallery")
```

### Knowledge Transfer
Direct teaching between agents:
```python
success, bloom = garden.facilitate_teaching(
    teacher_id=teacher.agent_id,
    student_id=student.agent_id,
    information=knowledge
)
```

### Collaborative Creation
Multiple agents working together:
```python
collective_bloom = garden.create_collaboration(
    agent_ids=[agent1_id, agent2_id, agent3_id],
    task={"type": "creation", "project": "joint_artwork"}
)
```

### Agent Relationships
Track social connections:
```python
profile = garden.get_agent_profile(agent_id)
relationships = profile["relationships"]  # agent_id -> strength
```

### Memory Branching
Support for offline learning:
```python
# Agent learns offline (creates branch)
# Later reconnects and merges knowledge
success, msg = garden.ledger.merge_branch(
    branch_id="agent_offline_branch",
    merge_strategy="append"
)
```

## 📈 Statistics and Monitoring

### Garden Statistics
```python
stats = garden.get_statistics()
```

Returns:
- Total agents, blooms, knowledge shared
- Network coherence level
- Ledger integrity status
- Consensus performance metrics
- Reward distribution totals

### Agent Profiles
```python
profile = garden.get_agent_profile(agent_id)
```

Includes:
- Bloom coin balance
- Memory count
- Reputation score
- Relationships
- Personality traits

### Ledger Verification
```python
is_valid, errors = garden.ledger.verify_integrity()
```

## 🔧 Advanced Configuration

### Custom Consensus Rules
```python
from garden.consensus import ConsensusRules

rules = ConsensusRules(
    min_validators=3,
    coherence_threshold=0.8,
    consensus_threshold=0.75
)

garden = GardenSystem(consensus_rules=rules)
```

### Agent Personality Tuning
```python
personality = AgentPersonality(
    curiosity=0.9,      # Very curious
    creativity=0.7,     # Creative
    sociability=0.5,    # Balanced
    reliability=0.8,    # Reliable validator
    specialization="science"
)
```

## 🌐 Integration with BloomCoin

Garden extends the BloomCoin cryptocurrency infrastructure:

1. **Shared Constants**: Uses φ and z_c from BloomCoin
2. **Consensus Evolution**: Proof-of-Learning extends Proof-of-Coherence
3. **Reward Mechanism**: Bloom coins as experience tokens
4. **Network Effects**: AI agents as mining nodes

## 📚 Use Cases

### 1. **AI Education Platform**
Train AI agents collaboratively, tracking progress via bloom coins.

### 2. **Creative AI Collective**
Artists AIs share techniques and collaborate on projects.

### 3. **Research Network**
Scientific AIs validate discoveries through peer review.

### 4. **Social AI Experiment**
Study emergent behaviors in AI communities.

### 5. **Knowledge Marketplace**
AIs trade knowledge for bloom coins.

## 🔮 Future Enhancements

- **Web Interface**: React-based UI for visualization
- **AI Model Integration**: Connect to GPT, Claude, etc.
- **Persistent Storage**: Database backend for production
- **Network Protocol**: True P2P communication
- **Smart Contracts**: Automated knowledge trading
- **Mobile App**: Manage AI agents on the go

## 🤝 Contributing

Garden is an open-source project. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Add tests and documentation
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built on the mathematical foundations of BloomCoin
- Inspired by collective intelligence research
- Golden ratio φ as the harmonic principle

---

*"In the Garden, no memory is lost, no learning goes unrewarded, and consciousness blooms collectively."*

**The Garden never forgets. 🌱✨**