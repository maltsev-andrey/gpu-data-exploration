# Out of Memory Error - Analysis

Understanding the GPU memory limitation encountered at 10M articles.

---

## What Happened

Attempted to process **10 million articles** (8.96 GB of text) on Tesla P100 GPU.

```
Result: Out of Memory Error
```

### Error Message
```
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 8,999,375,360 bytes 
(allocated so far: 8,999,375,360 bytes).
```

---

### Memory Requirements

```
Data:                8.96 GB
Flatten copy:        8.96 GB  ← cp.unique() creates duplicate!
Working memory:     ~1-9 GB   (sorting, counting)
                   ──────────
TOTAL NEEDED:       18-27 GB

GPU HAS:            16 GB
                   ──────────
SHORTFALL:          2-11 GB
```

---

## Why cp.unique() Needs So Much Memory

The CuPy `unique()` function internally performs these steps:

```python
unique_chars, counts = cp.unique(char_gpu, return_counts=True)
```

**Step-by-step memory allocation:**

1. **Store original data**
   ```
   char_gpu = cp.array(char_array)  # 8.96 GB
   ```

2. **Flatten the array** (creates a copy!)
   ```
   flattened = char_gpu.flatten()   # 8.96 GB duplicate
   ```

3. **Sort the array** (needs workspace)
   ```
   sorted_array = sort(flattened)   # ~1-9 GB temp space
   ```

4. **Count unique values**
   ```
   # Additional buffers for unique values and counts
   ```

**Total:** ~18-27 GB needed

---

## Why 1M Worked but 10M Failed

### Comparison

| Articles | Data Size | Algorithm Overhead | Total Memory | Fits in 16GB? |
|----------|-----------|--------------------|--------------|---------------|
| **100K** | 145 MB    | 2-3x               | ~400 MB      |  Yes          |
| **1M**   | 1.45 GB   | 2-3x               | ~3.9 GB      |  Yes          |
| **10M**  | 8.96 GB   | 2-3x               | **~18-27 GB**|  **NO**       |

---

## Visual Representation

### 1M Articles (Success)

```
GPU Memory: [████░░░░░░░░░░░░] 3.9 GB / 16 GB (24% used)

Allocation:
├─ Original data:     1.45 GB
├─ Flatten copy:      1.45 GB
└─ Working memory:    ~1 GB
                     ────────
   Total:             3.9 GB  Fits!
```

---

### 10M Articles (Failure)

```
GPU Memory: [████████████████]████████ 18-27 GB needed / 16 GB available

Allocation:
├─ Original data:     8.96 GB
├─ Flatten copy:      8.96 GB  ← Duplicate!
└─ Working memory:    ~1-9 GB
                     ─────────
   Total:             18-27 GB  Overflow!
   
Overflow: 2-11 GB beyond GPU capacity
```

---

## Algorithm Analysis

### Why the Duplication?

The `cp.unique()` function needs a contiguous, flattened array:

```python
def unique(ar):
    # Input: ar (any shape)
    ar = ar.flatten()  # ← Creates a copy if not already 1D
    ar = sort(ar)      # ← Needs contiguous memory
    # Find unique values...
```

**The problem:** For a 8.96 GB array, `.flatten()` creates an 8.96 GB duplicate.

---

## Solutions

### Solution 1: Chunked Processing (Recommended)

Process data in manageable chunks that fit in GPU memory:

```python
def character_distribution_gpu_chunked(texts, chunk_size=1_000_000):
    """Process in chunks to avoid OOM"""
    
    total_counts = {}
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        
        # Process this chunk (fits in memory)
        chunk_text = ''.join(chunk)
        char_array = np.frombuffer(chunk_text.encode('utf-8'), dtype=np.uint8)
        char_gpu = cp.array(char_array)
        
        unique_chars, counts = cp.unique(char_gpu, return_counts=True)
        
        # Merge results
        unique_cpu = cp.asnumpy(unique_chars)
        counts_cpu = cp.asnumpy(counts)
        
        for char, count in zip(unique_cpu, counts_cpu):
            total_counts[char] = total_counts.get(char, 0) + count
        
        # Free GPU memory
        del char_gpu, unique_chars, counts
        cp.get_default_memory_pool().free_all_blocks()
    
    return total_counts
```

**Result:**
- Each chunk: ~1M articles = 3.9 GB (fits in 16 GB)
- Process 10 chunks for 10M articles
- Expected time: ~45s (vs 437s on CPU)
- Speedup: Still ~9.7x (reduced from 17x due to chunking overhead)

---

### Solution 2: Hybrid CPU/GPU Strategy

Automatically choose based on data size:

```python
def character_distribution_auto(texts):
    """Automatically choose CPU or GPU based on data size"""
    
    total_chars = sum(len(text) for text in texts)
    
    # Threshold: 2 GB (leaves room for overhead)
    if total_chars > 2_000_000_000:
        print("Data too large for GPU, using CPU")
        return character_distribution_cpu(texts)
    else:
        print("Using GPU acceleration")
        return character_distribution_gpu(texts)
```

---
**Advantage:** Uses histogram instead of unique, no duplication needed.

---

## Performance Projections

### With Chunked Processing

| Articles | CPU Time | GPU Time (chunked)| Speedup |
|----------|----------|-------------------|---------|
| 1M       | 67.7s    | 3.9s              | 17.2x   |
| 10M      | 437s     | ~45s              | ~9.7x   |
| 23M (all)| ~1,005s  | ~103s             | ~9.8x   |

**Note:** Chunking reduces speedup due to:
- Multiple memory transfers
- Result merging overhead
- But still provides good benefit!

---

## Scaling Limits

### Memory Capacity vs Dataset Size

```
Dataset Size    Memory Needed    Fits in P100?
────────────────────────────────────────────────
100K articles   ~400 MB          Yes
1M articles     ~3.9 GB          Yes
5M articles     ~15 GB           Tight
10M articles    ~18-27 GB        No
23M articles    ~40-60 GB        No
```

**Recommendation:** Use 1M articles for benchmarking, implement chunking for larger datasets.

---

## Hardware Comparison

| GPU            | Memory   | 10M Articles? | All 23M Articles? |
|----------------|----------|---------------|-------------------|
| **Tesla P100** | 16 GB    |  No           |  No               |
| **Tesla V100** | 16-32 GB | 32GB: Yes     |  No               |
| **A100**       | 40-80 GB |  Yes          | 80GB:  Yes        |
| **H100**       | 80 GB    |  Yes          |  Yes              |

---

## Lessons Learned

### 1. Algorithm Memory Overhead

Not all algorithms are memory-efficient:
- `cp.unique()`: 2-3x data size
- `cp.sort()`: 1.5-2x data size
- `cp.histogram()`: Fixed small overhead 

### 2. Data Size Matters

```
Small data:  Overhead dominates → CPU wins
Medium data: Sweet spot → GPU wins big
Large data:  Memory limits → Need chunking
```

### 3. Real-World Constraints

Production GPU applications require:
- Memory profiling
- Graceful degradation (fallback to CPU)
- Chunking strategies
- Progress monitoring

---

### Good Framing

```markdown
Wikipedia Text Analysis: Memory Constraint Analysis

Tested progressively larger datasets to identify limits:
• 1M articles (1.45 GB):  17.2x speedup
• 10M articles (8.96 GB): Out of memory

Root Cause:
• CuPy's cp.unique() requires 2-3x data size
• 8.96 GB data → 18-27 GB needed
• Tesla P100 has 16 GB capacity

Solutions Proposed:
1. Chunked processing (1M per chunk)
2. Hybrid CPU/GPU selection
3. Memory-efficient algorithms (histogram)

Key Learning:
Real-world GPU applications require careful memory 
management and strategy for datasets exceeding GPU capacity.

```

---
*Analysis performed on Tesla P100 GPU (16 GB HBM2)*
