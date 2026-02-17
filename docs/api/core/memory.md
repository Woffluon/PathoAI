# core.memory - Memory Management

The memory module provides comprehensive memory monitoring, session management, and garbage collection to prevent out-of-memory errors in production.

## Overview

PathoAI implements proactive memory management to handle large histopathology images and deep learning models safely. The system monitors memory usage in real-time, manages user sessions, and performs automatic cleanup when memory pressure is detected.

## MemoryMonitor

**Class**: `core.memory.monitor.MemoryMonitor`

Real-time memory usage monitoring with configurable thresholds and automatic cleanup triggers.

### Constructor

```python
def __init__(self, config: MemoryConfig)
```

**Parameters**:
- `config` (MemoryConfig): Memory management configuration

**Example**:
```python
from core.memory import MemoryMonitor, MemoryConfig

config = MemoryConfig.from_env()
monitor = MemoryMonitor(config)
```

### Methods

#### get_current_metrics

```python
def get_current_metrics(self) -> MemoryMetrics
```

Get current memory usage metrics.

**Returns**:
- `metrics` (MemoryMetrics): Current memory statistics
  - `total_mb`: Total system memory (MB)
  - `available_mb`: Available memory (MB)
  - `used_mb`: Used memory (MB)
  - `percent_used`: Memory usage percentage
  - `timestamp`: Measurement timestamp

**Example**:
```python
metrics = monitor.get_current_metrics()
print(f"Memory usage: {metrics.percent_used:.1f}%")
print(f"Available: {metrics.available_mb:.0f} MB")
```

#### check_memory_available

```python
def check_memory_available(self, required_mb: float) -> bool
```

Check if sufficient memory is available for an operation.

**Parameters**:
- `required_mb` (float): Required memory in MB

**Returns**:
- `bool`: True if sufficient memory available

**Raises**:
- `InsufficientMemoryError`: If insufficient memory

**Example**:
```python
if monitor.check_memory_available(1500):
    # Proceed with operation
    result = run_inference(img)
else:
    # Handle insufficient memory
    print("Not enough memory")
```

## MemoryConfig

**Class**: `core.memory.config.MemoryConfig`

Configuration for memory management behavior.

### from_env

```python
@classmethod
def from_env(cls) -> MemoryConfig
```

Create configuration from environment variables.

**Environment Variables**:
- `MAX_MEMORY_PERCENT`: Memory warning threshold (default: 85)
- `MAX_SESSION_MEMORY_GB`: Per-session memory limit (default: 2.0)
- `SESSION_TIMEOUT_MINUTES`: Session timeout (default: 30)
- `AGGRESSIVE_GC`: Enable aggressive GC (default: false)

**Example**:
```python
from core.memory import MemoryConfig

config = MemoryConfig.from_env()
print(f"Memory threshold: {config.max_memory_percent}%")
```

## SessionManager

**Class**: `core.memory.session_manager.SessionManager`

Manages user sessions with automatic timeout and memory tracking.

### Constructor

```python
def __init__(
    self,
    timeout_minutes: int = 30,
    max_memory_gb: float = 2.0
)
```

**Parameters**:
- `timeout_minutes` (int): Session timeout in minutes
- `max_memory_gb` (float): Maximum memory per session (GB)

**Example**:
```python
from core.memory import SessionManager

manager = SessionManager(timeout_minutes=30, max_memory_gb=2.0)
```

### Methods

#### create_session

```python
def create_session(self) -> str
```

Create a new session.

**Returns**:
- `session_id` (str): Unique session identifier

**Example**:
```python
session_id = manager.create_session()
print(f"Created session: {session_id}")
```

#### update_session

```python
def update_session(self, session_id: str, **data: Any) -> None
```

Update session data.

**Parameters**:
- `session_id` (str): Session identifier
- `**data`: Key-value pairs to store

**Example**:
```python
manager.update_session(
    session_id,
    uploaded_image=img,
    analysis_results=results
)
```

#### get_session

```python
def get_session(self, session_id: str) -> Optional[SessionData]
```

Retrieve session data.

**Parameters**:
- `session_id` (str): Session identifier

**Returns**:
- `SessionData | None`: Session data or None if not found

**Example**:
```python
session = manager.get_session(session_id)
if session:
    print(f"Last access: {session.last_access}")
```

#### cleanup_inactive_sessions

```python
def cleanup_inactive_sessions(self) -> int
```

Remove inactive sessions.

**Returns**:
- `int`: Number of sessions cleaned up

**Example**:
```python
cleaned = manager.cleanup_inactive_sessions()
print(f"Cleaned up {cleaned} sessions")
```

## GarbageCollectionManager

**Class**: `core.memory.gc_manager.GarbageCollectionManager`

Manages Python garbage collection with configurable aggressiveness.

### Constructor

```python
def __init__(self, aggressiveness: str = "moderate")
```

**Parameters**:
- `aggressiveness` (str): GC level ("low", "moderate", "high")

**Aggressiveness Levels**:
- **low**: Minimal GC, best performance
- **moderate**: Balanced GC (default)
- **high**: Aggressive GC, best memory efficiency

**Example**:
```python
from core.memory import GarbageCollectionManager

gc_manager = GarbageCollectionManager(aggressiveness="moderate")
gc_manager.configure_gc()
```

### Methods

#### collect_with_stats

```python
def collect_with_stats(self, generation: int = 2) -> Dict[str, Any]
```

Perform garbage collection and return statistics.

**Parameters**:
- `generation` (int): GC generation (0, 1, or 2)

**Returns**:
- `stats` (dict): Collection statistics
  - `collected`: Objects collected
  - `uncollectable`: Uncollectable objects
  - `duration_ms`: Collection duration

**Example**:
```python
stats = gc_manager.collect_with_stats(generation=2)
print(f"Collected {stats['collected']} objects in {stats['duration_ms']:.1f}ms")
```

## Memory Management Workflow

**Example**: Complete memory management workflow

```python
from core.memory import (
    MemoryMonitor,
    MemoryConfig,
    SessionManager,
    GarbageCollectionManager
)

# 1. Initialize components
config = MemoryConfig.from_env()
monitor = MemoryMonitor(config)
session_mgr = SessionManager(timeout_minutes=30)
gc_mgr = GarbageCollectionManager(aggressiveness="moderate")

# 2. Create session
session_id = session_mgr.create_session()

# 3. Check memory before operation
if monitor.check_memory_available(required_mb=1500):
    # 4. Perform operation
    result = run_inference(img)
    
    # 5. Store results
    session_mgr.update_session(session_id, results=result)
    
    # 6. Cleanup after operation
    stats = gc_mgr.collect_with_stats()
    print(f"Freed {stats['collected']} objects")
else:
    print("Insufficient memory")

# 7. Periodic cleanup
cleaned = session_mgr.cleanup_inactive_sessions()
print(f"Cleaned {cleaned} inactive sessions")
```

## Memory Monitoring

**Example**: Continuous memory monitoring

```python
import time
from core.memory import MemoryMonitor, MemoryConfig

config = MemoryConfig.from_env()
monitor = MemoryMonitor(config)

while True:
    metrics = monitor.get_current_metrics()
    
    if metrics.percent_used > 85:
        print(f"WARNING: High memory usage: {metrics.percent_used:.1f}%")
        # Trigger cleanup
        gc_mgr.collect_with_stats()
    
    time.sleep(60)  # Check every minute
```

## Configuration

**Memory Management Defaults** (from `core.defaults.MemoryManagementDefaults`):

```python
# Session Management
session_timeout_minutes = 30
max_session_memory_gb = 2.0
session_cleanup_interval_minutes = 5

# Memory Monitoring
memory_warning_threshold = 85.0  # percent
monitoring_interval_sec = 1.0
enable_background_monitoring = False

# Garbage Collection
gc_aggressiveness = "moderate"
enable_auto_gc = True
gc_threshold_gen0 = 700
gc_threshold_gen1 = 10
gc_threshold_gen2 = 10

# Emergency Cleanup
emergency_cleanup_threshold_mb = 500.0
enable_proactive_gc = True
proactive_gc_threshold = 90.0
enable_request_rejection = True
```

## Performance Impact

**Memory Monitoring Overhead**:
- CPU: < 1% (sampling every 1 second)
- Memory: ~10 MB for monitoring data structures

**Garbage Collection Impact**:
- **Low**: ~50ms per collection, minimal impact
- **Moderate**: ~100ms per collection, acceptable
- **High**: ~200ms per collection, noticeable pauses

**Session Management Overhead**:
- Memory: ~1-5 MB per session
- CPU: Negligible (cleanup runs every 5 minutes)

## See Also

- [core.inference](inference.md): Model inference with memory cleanup
- [core.exceptions](exceptions.md): Memory-related exceptions
- [config.settings](../config/settings.md): Configuration

---

*Source files: `core/memory/monitor.py`, `core/memory/session_manager.py`, `core/memory/gc_manager.py`, `core/memory/config.py`*
