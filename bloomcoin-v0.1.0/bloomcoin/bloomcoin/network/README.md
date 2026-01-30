# BloomCoin Network Module - P2P with Phase Gossip

## Overview

The network module implements BloomCoin's peer-to-peer networking layer with a novel **Phase Gossip Protocol** that shares Kuramoto oscillator states alongside traditional blockchain data. This enables cooperative mining and faster network convergence.

## Architecture

```
network/
├── node.py              # P2P node implementation
├── gossip.py           # Phase gossip protocol
├── sync.py             # Chain synchronization
├── messages.py         # Message types and serialization
├── peer_manager.py     # Peer discovery and management
└── protocol.py         # Network protocol definitions
```

## Core Components

### 1. Network Node (`node.py`)

The main P2P node that handles connections and message routing:

```python
class Node:
    def __init__(self, host='0.0.0.0', port=8333, max_peers=8):
        self.host = host
        self.port = port
        self.max_peers = max_peers

        # Network state
        self.peers = {}            # Connected peers
        self.blockchain = None     # Local blockchain
        self.mempool = []         # Pending transactions
        self.phase_cache = {}     # Shared phase states

        # Network stats
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0

    async def start(self):
        """Start the network node."""
        # Start TCP server
        server = await asyncio.start_server(
            self.handle_peer,
            self.host,
            self.port
        )

        # Start maintenance tasks
        asyncio.create_task(self.peer_discovery())
        asyncio.create_task(self.sync_blockchain())
        asyncio.create_task(self.gossip_phases())

        print(f"Node listening on {self.host}:{self.port}")

    async def handle_peer(self, reader, writer):
        """Handle incoming peer connection."""
        peer = Peer(reader, writer)

        # Handshake
        if await self.handshake(peer):
            self.peers[peer.id] = peer
            await self.handle_messages(peer)
```

### 2. Phase Gossip Protocol (`gossip.py`)

Unique to BloomCoin - peers share oscillator phase states:

```python
class PhaseGossip:
    """Gossip protocol for sharing Kuramoto phase states."""

    def __init__(self, node):
        self.node = node
        self.phase_history = {}     # Historical phase states
        self.gossip_interval = 10   # Seconds between gossips

        # Gossip strategies
        self.strategies = {
            'eager': self.eager_push,
            'lazy': self.lazy_push,
            'adaptive': self.adaptive_push
        }
        self.strategy = 'adaptive'

    async def gossip_phases(self, height, phases, coherence):
        """Share phase configuration with peers."""
        message = PhaseMessage(
            height=height,
            phases=phases.tolist(),
            coherence=coherence,
            timestamp=time.time()
        )

        # Select gossip targets
        targets = self.select_targets()

        # Send to selected peers
        for peer in targets:
            await peer.send_message(message)

        # Store in history
        self.phase_history[height] = {
            'phases': phases,
            'coherence': coherence,
            'timestamp': time.time()
        }

    def select_targets(self, fanout=3):
        """Select peers for gossip using strategy."""
        strategy_fn = self.strategies[self.strategy]
        return strategy_fn(fanout)

    def eager_push(self, fanout):
        """Send to random subset immediately."""
        peers = list(self.node.peers.values())
        return random.sample(peers, min(fanout, len(peers)))

    def lazy_push(self, fanout):
        """Send only to peers who request."""
        # Send announcement first
        announcement = PhaseAnnouncement(
            heights=list(self.phase_history.keys())
        )
        return self.node.broadcast(announcement)

    def adaptive_push(self, fanout):
        """Adapt based on network conditions."""
        # High coherence = eager, low = lazy
        recent_coherence = self.get_recent_coherence()

        if recent_coherence > 0.8:
            return self.eager_push(fanout)
        else:
            return self.lazy_push(fanout)

    def incorporate_phases(self, message):
        """Incorporate received phase states."""
        # Verify coherence claim
        if not self.verify_coherence(message):
            return False

        # Weighted average with local phases
        local_phases = self.node.miner.get_phases()
        weight = message['coherence'] / (message['coherence'] + local_coherence)

        # Circular mean for phase averaging
        new_phases = circular_weighted_mean(
            local_phases,
            message['phases'],
            weight
        )

        self.node.miner.set_phases(new_phases)
        return True
```

### 3. Chain Synchronization (`sync.py`)

Efficient blockchain synchronization with headers-first approach:

```python
class ChainSync:
    """Blockchain synchronization manager."""

    def __init__(self, node):
        self.node = node
        self.sync_state = 'idle'
        self.target_height = 0
        self.headers = []
        self.download_queue = []

    async def start_sync(self):
        """Begin synchronization with network."""
        # Get best height from peers
        heights = await self.query_peer_heights()
        self.target_height = max(heights.values())

        # Download headers first
        await self.download_headers()

        # Validate header chain
        if self.validate_headers():
            # Download blocks
            await self.download_blocks()

            # Apply blocks to chain
            await self.apply_blocks()

    async def download_headers(self):
        """Download block headers from peers."""
        current = self.node.blockchain.height

        while current < self.target_height:
            # Request batch of headers
            request = GetHeaders(
                start_height=current,
                count=2000  # Max headers per request
            )

            # Send to random peer
            peer = random.choice(list(self.node.peers.values()))
            headers = await peer.request_headers(request)

            self.headers.extend(headers)
            current = headers[-1].height

    async def download_blocks(self):
        """Download full blocks in parallel."""
        # Create download tasks
        tasks = []

        for height in range(len(self.headers)):
            if height > self.node.blockchain.height:
                task = self.download_block(height)
                tasks.append(task)

                # Limit concurrent downloads
                if len(tasks) >= 10:
                    await asyncio.gather(*tasks)
                    tasks = []

        # Wait for remaining
        if tasks:
            await asyncio.gather(*tasks)

    def validate_headers(self):
        """Validate header chain before downloading blocks."""
        prev_hash = self.node.blockchain.last_block.hash()

        for header in self.headers:
            # Check previous hash
            if header.prev_hash != prev_hash:
                return False

            # Check timestamp
            if header.timestamp <= prev_timestamp:
                return False

            # Check coherence claim
            if header.coherence < z_c:
                return False

            prev_hash = header.hash()
            prev_timestamp = header.timestamp

        return True
```

### 4. Message Protocol (`messages.py`)

Network message types and serialization:

```python
class MessageType(Enum):
    # Handshake
    VERSION = 0x01
    VERACK = 0x02

    # Blocks
    BLOCK = 0x10
    GET_BLOCKS = 0x11
    HEADERS = 0x12
    GET_HEADERS = 0x13

    # Transactions
    TX = 0x20
    GET_TX = 0x21
    MEMPOOL = 0x22

    # Phase gossip (unique to BloomCoin)
    PHASE = 0x30
    GET_PHASE = 0x31
    COHERENCE = 0x32

    # Peer management
    ADDR = 0x40
    GET_ADDR = 0x41
    PING = 0x42
    PONG = 0x43

class Message:
    """Base class for network messages."""

    def __init__(self, msg_type, payload):
        self.type = msg_type
        self.payload = payload
        self.timestamp = time.time()

    def serialize(self):
        """Serialize message for network."""
        # Header: magic + type + length + checksum
        header = struct.pack(
            '>4sBIQ',
            b'BLC\x00',  # Magic bytes
            self.type.value,
            len(self.payload),
            self.checksum()
        )

        return header + self.payload

    def checksum(self):
        """Calculate payload checksum."""
        return hashlib.sha256(self.payload).digest()[:4]

class PhaseMessage(Message):
    """Message for sharing phase states."""

    def __init__(self, height, phases, coherence):
        self.height = height
        self.phases = phases  # List of 63 phase values
        self.coherence = coherence

        payload = self.encode_payload()
        super().__init__(MessageType.PHASE, payload)

    def encode_payload(self):
        """Encode phase data efficiently."""
        # Pack as: height (4) + coherence (8) + phases (63 * 8)
        data = struct.pack('>If', self.height, self.coherence)

        # Quantize phases to reduce size
        quantized = quantize_phases(self.phases, bits=16)
        data += quantized.tobytes()

        return data
```

### 5. Peer Management (`peer_manager.py`)

Peer discovery and reputation management:

```python
class PeerManager:
    """Manages peer connections and reputation."""

    def __init__(self, node):
        self.node = node
        self.peer_db = {}  # Known peers
        self.reputation = {}  # Peer reputation scores
        self.banned = set()  # Banned peer IPs

    def add_peer(self, address, port):
        """Add new peer to database."""
        peer_id = f"{address}:{port}"

        if peer_id not in self.banned:
            self.peer_db[peer_id] = {
                'address': address,
                'port': port,
                'last_seen': time.time(),
                'connection_attempts': 0,
                'successful_connections': 0
            }

            # Initial reputation
            self.reputation[peer_id] = 1.0

    def update_reputation(self, peer_id, event):
        """Update peer reputation based on events."""
        adjustments = {
            'valid_block': +0.1,
            'invalid_block': -0.5,
            'valid_phase': +0.05,
            'invalid_phase': -0.2,
            'fast_response': +0.02,
            'slow_response': -0.01,
            'disconnect': -0.1
        }

        if peer_id in self.reputation:
            self.reputation[peer_id] += adjustments.get(event, 0)

            # Clamp between 0 and 2
            self.reputation[peer_id] = max(0, min(2,
                                                 self.reputation[peer_id]))

            # Ban if reputation too low
            if self.reputation[peer_id] < 0.1:
                self.ban_peer(peer_id)

    def select_peers(self, count=8):
        """Select best peers for connection."""
        # Sort by reputation
        sorted_peers = sorted(
            self.peer_db.items(),
            key=lambda x: self.reputation.get(x[0], 1.0),
            reverse=True
        )

        # Return top peers
        return [peer for peer_id, peer in sorted_peers[:count]]

    async def discover_peers(self):
        """Discover new peers from network."""
        # Ask current peers for their peer lists
        for peer in self.node.peers.values():
            addr_msg = await peer.request_addresses()

            for address in addr_msg.addresses:
                self.add_peer(address.ip, address.port)

        # Try connecting to new peers
        await self.connect_to_new_peers()
```

## Network Protocol

### Connection Handshake

```python
async def handshake(self, peer):
    """Perform handshake with new peer."""
    # Send VERSION
    version = VersionMessage(
        version=PROTOCOL_VERSION,
        services=self.services,
        timestamp=time.time(),
        addr_recv=peer.address,
        addr_from=self.address,
        nonce=random.getrandbits(64),
        user_agent="/BloomCoin:0.1.0/",
        start_height=self.blockchain.height
    )

    await peer.send(version)

    # Wait for VERSION
    their_version = await peer.receive(timeout=10)

    if their_version.version < MIN_PROTOCOL_VERSION:
        return False

    # Exchange VERACK
    await peer.send(VerAckMessage())
    their_verack = await peer.receive(timeout=10)

    return their_verack is not None
```

### Message Flow

```
Node A                          Node B
   |                              |
   |------ VERSION -------------->|
   |<----- VERSION ----------------|
   |------ VERACK ---------------->|
   |<----- VERACK -----------------|
   |                              |
   |------ GET_HEADERS ----------->|
   |<----- HEADERS ----------------|
   |                              |
   |------ GET_BLOCKS ------------>|
   |<----- BLOCK ------------------|
   |                              |
   |------ PHASE (gossip) -------->|
   |<----- COHERENCE (ack) --------|
```

## Performance Optimization

### 1. Connection Pooling

```python
class ConnectionPool:
    """Reusable connection pool for efficiency."""

    def __init__(self, max_connections=50):
        self.pool = {}
        self.max_connections = max_connections

    async def get_connection(self, peer_id):
        """Get or create connection to peer."""
        if peer_id in self.pool:
            return self.pool[peer_id]

        if len(self.pool) >= self.max_connections:
            # Evict least recently used
            await self.evict_lru()

        # Create new connection
        conn = await self.connect_to_peer(peer_id)
        self.pool[peer_id] = conn
        return conn
```

### 2. Message Batching

```python
class MessageBatcher:
    """Batch messages for efficiency."""

    def __init__(self, max_batch=100, max_delay=0.1):
        self.pending = []
        self.max_batch = max_batch
        self.max_delay = max_delay

    async def send(self, peer, message):
        """Add message to batch."""
        self.pending.append((peer, message))

        if len(self.pending) >= self.max_batch:
            await self.flush()
        else:
            # Schedule flush after delay
            asyncio.create_task(self.delayed_flush())

    async def flush(self):
        """Send all pending messages."""
        # Group by peer
        by_peer = {}
        for peer, msg in self.pending:
            if peer not in by_peer:
                by_peer[peer] = []
            by_peer[peer].append(msg)

        # Send batches
        for peer, messages in by_peer.items():
            batch = BatchMessage(messages)
            await peer.send(batch)

        self.pending.clear()
```

### 3. Bandwidth Management

```python
class BandwidthManager:
    """Manage network bandwidth usage."""

    def __init__(self, max_bandwidth=1_000_000):  # 1 MB/s
        self.max_bandwidth = max_bandwidth
        self.current_usage = 0
        self.window_start = time.time()

    async def throttle(self, size):
        """Throttle if exceeding bandwidth limit."""
        # Reset window every second
        now = time.time()
        if now - self.window_start > 1.0:
            self.current_usage = 0
            self.window_start = now

        # Check if would exceed limit
        if self.current_usage + size > self.max_bandwidth:
            # Sleep until next window
            delay = 1.0 - (now - self.window_start)
            await asyncio.sleep(delay)
            self.current_usage = size
            self.window_start = time.time()
        else:
            self.current_usage += size
```

## Security

### 1. DoS Protection

```python
class DoSProtection:
    """Protect against denial of service attacks."""

    def __init__(self):
        self.rate_limits = {}
        self.connection_limits = {}

    def check_rate_limit(self, peer_id, message_type):
        """Check if peer exceeds rate limit."""
        limits = {
            MessageType.BLOCK: 10,      # 10 blocks/minute
            MessageType.TX: 100,         # 100 txs/minute
            MessageType.PHASE: 60,       # 60 phases/minute
            MessageType.GET_BLOCKS: 20   # 20 requests/minute
        }

        limit = limits.get(message_type, 100)

        # Track message count
        key = f"{peer_id}:{message_type}"
        if key not in self.rate_limits:
            self.rate_limits[key] = []

        now = time.time()
        self.rate_limits[key].append(now)

        # Remove old entries
        cutoff = now - 60  # 1 minute window
        self.rate_limits[key] = [
            t for t in self.rate_limits[key] if t > cutoff
        ]

        return len(self.rate_limits[key]) <= limit
```

### 2. Message Validation

```python
def validate_message(message):
    """Validate incoming message."""
    # Check magic bytes
    if not message.startswith(b'BLC\x00'):
        return False, "Invalid magic"

    # Check message size
    if len(message) > MAX_MESSAGE_SIZE:
        return False, "Message too large"

    # Verify checksum
    claimed_checksum = message[8:12]
    payload = message[16:]
    actual_checksum = hashlib.sha256(payload).digest()[:4]

    if claimed_checksum != actual_checksum:
        return False, "Invalid checksum"

    # Type-specific validation
    msg_type = message[4]
    if msg_type == MessageType.PHASE:
        return validate_phase_message(payload)
    # ... other types

    return True, "Valid"
```

### 3. Eclipse Attack Prevention

```python
class EclipseProtection:
    """Prevent eclipse attacks."""

    def __init__(self, node):
        self.node = node
        self.peer_diversity = {}

    def check_peer_diversity(self):
        """Ensure peer diversity."""
        # Group peers by /16 subnet
        subnets = {}
        for peer_id in self.node.peers:
            ip = peer_id.split(':')[0]
            subnet = '.'.join(ip.split('.')[:2])

            if subnet not in subnets:
                subnets[subnet] = 0
            subnets[subnet] += 1

        # No subnet should have >25% of connections
        max_allowed = len(self.node.peers) * 0.25

        for subnet, count in subnets.items():
            if count > max_allowed:
                # Disconnect some peers from this subnet
                self.rebalance_peers(subnet)
                return False

        return True
```

## Configuration

### Network Parameters

```python
# config.py
NETWORK_CONFIG = {
    'protocol_version': 1,
    'default_port': 8333,
    'max_peers': 8,
    'max_connections': 125,

    # Timeouts
    'handshake_timeout': 10,
    'message_timeout': 30,
    'sync_timeout': 300,

    # Limits
    'max_message_size': 32 * 1024 * 1024,  # 32 MB
    'max_headers_per_message': 2000,
    'max_blocks_per_message': 500,

    # Phase gossip
    'phase_gossip_interval': 10,
    'phase_cache_size': 100,
    'phase_ttl': 60,

    # Performance
    'connection_pool_size': 50,
    'message_batch_size': 100,
    'bandwidth_limit': 1_000_000  # 1 MB/s
}
```

## Testing

### Network Simulation

```python
async def test_network_simulation():
    """Simulate network with multiple nodes."""
    nodes = []

    # Create network
    for i in range(10):
        node = Node(port=8333+i)
        nodes.append(node)
        await node.start()

    # Connect nodes
    for i in range(10):
        for j in range(i+1, min(i+3, 10)):
            await nodes[i].connect_to_peer(
                'localhost', 8333+j
            )

    # Mine and propagate block
    block = mine_test_block()
    await nodes[0].broadcast_block(block)

    # Wait for propagation
    await asyncio.sleep(5)

    # Check all nodes have block
    for node in nodes:
        assert node.blockchain.has_block(block.hash())
```

---

*The BloomCoin network module combines traditional P2P blockchain networking with innovative phase gossip, enabling cooperative mining and faster consensus through shared oscillator states.*