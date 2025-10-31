# GPU Data Processing - Dataset Exploration

Exploration scripts for validating large-scale datasets used in GPU-accelerated computing benchmarks.

## Project Purpose

This project contains data integrity verification tools for two massive real-world datasets:
- **Wikipedia**: 23 million article sections (SQLite database)
- **GBSense HDF5**: Industrial sensor time-series data (8.8 GB across 4 files)

These datasets will be used for GPU vs CPU performance benchmarking on a Tesla P100 GPU.

## Datasets

### 1. Wikipedia Articles Database
- **File**:   `enwiki-20170820.db` (21.4 GB)
- **Source**: English Wikipedia dump from August 2017
- **Format**: SQLite database
- **Structure**:
  ```
  Table: ARTICLES
  Rows: 23,046,187
  Columns:
    - ARTICLE_ID (INTEGER)
    - TITLE (TEXT)
    - SECTION_TITLE (TEXT)
    - SECTION_TEXT (TEXT)
  ```

### 2. GBSense HDF5 Sensor Data
- **Total Size**: 8.8 GB (4 files)
- **Format**:     HDF5 (Hierarchical Data Format)
- **Purpose**:    Industrial/scientific sensor measurements

#### data_1_train.h5
```
X: (124,800, 1,024, 16) - 4.09 GB
   ├─ 124,800 samples
   ├─ 1,024 sensors per sample
   └─ 16 time steps per sensor
   
Y: (124,800, 1) - 124 KB
   └─ Binary classification labels

Data type: int16
Range:     -8,192 to 7,717
Mean:      15.08
Task:      Binary classification (anomaly detection)
```

#### data_2_train.h5
```
X: (102,400, 1,024, 16) - 3.36 GB
   ├─ 102,400 samples
   ├─ 1,024 sensors per sample
   └─ 16 time steps per sensor

Y: (102,400, 24) - 2.46 MB
   └─ Multi-output prediction (24 values)

Data type: int16
Range:     -14,053 to 14,182
Mean:      23.39
Task:      Multi-label prediction
```

## Tools

### `explore_wiki.py`
Analyzes SQLite database structure and integrity.

**Features:**
- Table discovery and schema analysis
- Row counting
- Sample data extraction
- Column metadata inspection

**Usage:**
```bash
python3 explore_wiki.py 
# Default path: /home/ansible/gpu-projects/shared-data/wiki/enwiki-20170820.db
```

**Output:**
- Database structure
- Table schemas
- Row counts
- Sample records (first 3 rows)
- Column types and constraints

### `explore_h5.py`
Analyzes HDF5 file structure and contents.

**Features:**
- Dataset discovery and shape analysis
- Data type inspection
- Memory size calculation
- Statistical summaries (min, max, mean)
- Sample data visualization (for 1D, 2D, 3D arrays)

**Usage:**
```bash
python3 explore_h5.py <path_to_h5_file>
```

**Example:**
```bash
python3 explore_h5.py /path/to/data_1_train.h5
```

**Output:**
- Root keys
- Dataset shapes and dimensions
- Memory requirements
- Data ranges and statistics
- Sample slices (for multi-dimensional arrays)

## Key Findings

### Wikipedia Dataset
- **Scale**:          23+ million text sections
- **Content**:        Article text from 5+ million Wikipedia pages
- **Size**:           ~10-20 GB of text data
- **Use Case**:       Text processing, NLP, word frequency analysis
- **GPU Operations**: String processing, character analysis, pattern matching

### HDF5 Sensor Data

#### Dataset Comparison

| Metric      | data_1_train | data_2_train |
|-------------|--------------|--------------|
| Samples     | 124,800      | 102,400      |
| Features    | 1,024 × 16   | 1,024 × 16   |
| X Size      | 4.09 GB      | 3.36 GB      |
| Elements    | 2.0 billion  | 1.6 billion  |
| Y Structure | (124800, 1)  | (102400, 24) |
| Task Type   | Binary       | Multi-output |

#### Data Interpretation

**X Data (Sensor Readings):**
- 3D time-series structure: `(samples, sensors, timesteps)`
- Likely industrial sensor monitoring or signal processing
- Possible domains: vibration analysis, acoustic sensing, IoT measurements

**Y Data (Labels):**
- `data_1`: Single binary label → Classification task
- `data_2`: 24 outputs → Sequential or multi-target prediction

## Next Steps

1. **GPU Benchmarking**:       Statistical analysis, FFT, matrix operations
2. **Performance Comparison**: CPU vs GPU speedup measurements
3. **Optimization**:           Memory-efficient processing for large datasets

### Expected GPU Speedups (Tesla P100)

| Operation              | Expected Speedup |
|------------------------|------------------|
| Statistical Analysis   | 60-80x           |
| FFT (Frequency Domain) | 40-60x           |
| Normalization          | 70-90x           |
| Matrix Operations      | 100-300x         |

## Environment

- **GPU**:           Tesla P100 (Pascal architecture, 16 GB HBM2)
- **OS**:            RHEL 9
- **Python**:        3.9+
- **Key Libraries**: numpy, h5py, sqlite3

## Project Structure

```
03-gpu-data-processing/
├── src/
│   ├── wikipedia/
│   │   └── explore_wiki.py
│   └── hdf5/
│       └── explore_h5.py
├── data/
│   ├── wikipedia/ (symlink to shared-data)
│   └── hdf5/ (symlink to shared-data)
├── benchmarks/ (future: GPU benchmark results)
└── docs/
```

## Security

- Root user detection built-in
- Prevents accidental GPU operations as root
- Data integrity validation before processing

## Sample Output

### Wikipedia Exploration
```
============================
Database Structure Analysis
============================

Found 1 table(s):

TABLE: ARTICLES
Columns:
  [0] ARTICLE_ID           INTEGER         NULL    
  [1] TITLE                TEXT            NULL    
  [2] SECTION_TITLE        TEXT            NULL    
  [3] SECTION_TEXT         TEXT            NULL    

Total rows: 23,046,187
```

### HDF5 Exploration
```
============================
ANALYZING: data_1_train.h5
============================

Root keys: ['X', 'Y']

DATASET: X
  Shape:  (124800, 1024, 16)
  Dtype:  int16
  Size:   2,044,723,200 elements
  Memory: 4.09 GB
  Stats:  min=-8192, max=7717, mean=15.083942

DATASET: Y
  Shape:  (124800, 1)
  Dtype:  uint8
  Memory: 124.80 KB
```

## Portfolio Context

This is part of a larger GPU computing portfolio demonstrating:

1. **Data Preparation**:     Validation of multi-GB datasets
2. **Domain Versatility**:   Text (Wikipedia) + Scientific (Sensors)
3. **Scale Handling**:       Billions of data points
4. **Production Readiness**: Error handling, memory efficiency

Related projects:
- Monte Carlo Option Pricing (25,476x GPU speedup)
- GPU-accelerated text analysis
- Time-series sensor processing

## License

MIT License

## Author

Developed as part of GPU computing portfolio for engineering applications.

---

**Status**: ✅ Data integrity verified, ready for GPU benchmarking
