#!/usr/bin/env python3
"""
GPU-Accelerated Wikipedia Text Analysis
Compares CPU vs GPU performance for text processing tasks

Computations:
1. Word frequency analysis across all articles
2. Text length statistics
3. Character distribution analysis
4. Search pattern matching

Requirements:
- CuPy for GPU acceleration
- NumPy for CPU baseline
- SQLite3 for database access
"""

import os
import sys
import time
import sqlite3
import re
from collections import Counter
import numpy as np

# Check if user roor or not
if os.getuid() == 0:
    print("\n"+"-"*60 )
    print("ERROR: Don't run GPU as root! Use: su - ansible")
    print("-"*60 + "\n")
    sys.exit(1)
    
# Patch NumPy for scikit-cuda compatibility
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'complex'):
    np.complex = np.complex128

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy available - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - GPU computations will be skipped")

class WikipediaAnalyzer:
    """Analyzes Wikipedia text data with CPU and GPU implementations"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f" Connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f" Database connection error: {e}")
            return False

    def get_table_info(self):
        """Get information about database tables"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print("\n" + "="*60)
        print("DATABASE TABLES")
        print("="*60 )

        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count:,} rows")
        
        return tables

    def load_text_data(self, limit=None):
        """Load text data from database
        
        Args:
            limit: Maximum number of articles to load (None for all)
        
        Returns:
            List of text strings
        """
        cursor = self.conn.cursor()

        # Query for the ARTICLES table with SECTION_TEXT column
        query = "SELECT SECTION_TEXT FROM ARTICLES WHERE SECTION_TEXT IS NOT NULL"
        if limit:
            query += f" Limit {limit}"

        print(f"\nExecuting query: {query}")

        try:
            cursor.execute(query)
            texts = [row[0] for row in cursor.fetchall() if row[0]]
        except sqlite3.Error as e:
            print(f" Query failed: {e}")
            print("\nTrying fallback queries...")

            # Fallback queries
            possible_queries = [
                "SELECT SECTION_TEXT FROM articles",
                "SELECT text FROM articles", 
                "SELECT content FROM articles",
                "SELECT body FROM articles",
            ]

            texts = []
            for fallback_query in possible_queries:
                try:
                    query = fallback_query
                    if limit:
                        query += f" LIMIT {limit}"

                        print(f" Trying: {query}")
                        cursor.execute(query)
                        texts = [row[0] for row in cursor.fetchall() if row[0]]

                        if texts:
                            break
                except sqlite3.Error:
                    continue

            if not texts:
                print(" No text data found")
                return [], None

        print(f"\n Loaded {len(texts):,} text entries")
        if texts:
            total_chars = sum(len(t) for t in texts)
            print(f" Total characters: {total_chars:,}")
            print(f" Average length: {total_chars / len(texts):.1f} chars")

        return texts, query

    def analyze_text_lengths_cpu(self, texts):
        """CPU: Calculate text length statistics"""
        print("\n" + "="*60)
        print("CPU: Text Length Analysis")
        print("="*60)

        start = time.time()

        # Convert to numpy array
        lengths = np.array([len(text) for text in texts], dtype=np.int32)

        # Calculate statistics
        stats = {
            'count': len(lengths),
            'total': np.sum(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths)
        }
        
        end = time.time()
        elapsed = end - start

        print(f"\nResults:")
        print(f" Articles: {stats['count']:,}")
        print(f"  Total chars: {stats['total']:,}")
        print(f"  Mean length: {stats['mean']:.1f}")
        print(f"  Std dev: {stats['std']:.1f}")
        print(f"  Min: {stats['min']:,}")
        print(f"  Max: {stats['max']:,}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"\n  CPU Time: {elapsed:.6f} seconds")
        
        return stats, elapsed

    def analyte_text_lengths_gpu(self, texts):
        """GPU: Calculate text length statistics"""
        if not GPU_AVAILABLE:
            print("\n GPU not available")
            return None, 0
        
        print("\n" + "="*80)
        print("GPU: Text Length Analysis")
        print("="*80)
        
        start = time.time()

        # Transfer data to GPU
        lengths_cpu = np.array([len(text) for text in texts], dtype=np.int32)
        lengths_gpu = cp.array(lengths_cpu)

        # Calculate statistics on GPU
        stats = {
            'count': int(lengths_gpu.size),
            'total': int(cp.sum(lengths_gpu)),
            'mean': float(cp.mean(lengths_gpu)),
            'std': float(cp.std(lengths_gpu)),
            'min': int(cp.min(lengths_gpu)),
            'max': int(cp.max(lengths_gpu)),
            'median': float(cp.median(lengths_gpu))
        }
        
        # Ensure computation is complete
        cp.cuda.Stream.null.synchronize()

        end = time.time()
        elapsed = end - start

        print(f"\nResults:")
        print(f"  Articles: {stats['count']:,}")
        print(f"  Total chars: {stats['total']:,}")
        print(f"  Mean length: {stats['mean']:.1f}")
        print(f"  Std dev: {stats['std']:.1f}")
        print(f"  Min: {stats['min']:,}")
        print(f"  Max: {stats['max']:,}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"\n  GPU Time: {elapsed:.6f} seconds")
        
        return stats, elapsed

    def word_frequency_cpu(self, texts, top_n=100):
        """CPU: Count word frequencies"""
        print("\n" + "="*60)
        print(f"CPU: Word Frequency Analysis (Top {top_n})")
        print("="*60)

        start = time.time()

        # Tokenize and count
        word_counts = Counter()
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)

        top_words = word_counts.most_common(top_n)
        total_words = sum(word_counts.values())
        unique_words = len(word_counts)

        end = time.time()
        elapsed = end - start

        print(f"\nResults:")
        print(f" Total words: {total_words:,}")
        print(f"  Unique words: {unique_words:,}")
        print(f"\n  Top {min(10, len(top_words))} words:")
        for word, count in top_words[:10]:
            print(f"    {word:20s}: {count:,}")
        
        print(f"\n CPU Time: {elapsed:.6f} seconds")
        
        return word_counts, elapsed

    def character_distribution_cpu(self, texts):
        """CPU: Analyze character distribution"""
        print("\n" + "="*60)
        print("CPU: Character Distribution Analysis")
        print("="*60)
        
        start = time.time()

        # Concatenate all text
        all_text = ''.join(texts)  
        total_chars = len(all_text)

        # Count characters frequencies
        char_counts = Counter(all_text)

        # Calculate statistics
        letter_count = sum(count for char, count in char_counts.items() if char.isalpha())
        digit_count = sum(count for char, count in char_counts.items() if char.isdigit())
        space_count = char_counts.get(' ', 0)
        
        end = time.time()
        elapsed = end - start
        
        print(f"\nResults:")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Unique characters: {len(char_counts):,}")
        print(f"  Letters: {letter_count:,} ({100*letter_count/total_chars:.1f}%)")
        print(f"  Digits: {digit_count:,} ({100*digit_count/total_chars:.1f}%)")
        print(f"  Spaces: {space_count:,} ({100*space_count/total_chars:.1f}%)")
        
        print(f"\n CPU Time: {elapsed:.6f} seconds")
        
        return char_counts, elapsed

    def character_distribution_gpu(self, texts):
        """GPU: Analyze character distribution"""
        if not GPU_AVAILABLE:
            print("\n GPU not available")
            return None, 0
        
        print("\n" + "="*60)
        print("GPU: Character Distribution Analysis")
        print("="*60)
        
        start = time.time()

        # Concatenate and convert to numeric array
        all_text = ''.join(texts)
        char_array = np.frombuffer(all_text.encode('utf-8'), dtype=np.uint8)
        
        # Transfer to GPU
        char_gpu = cp.array(char_array)
        
        # Count unique values
        unique_chars, counts = cp.unique(char_gpu, return_counts=True)
        
        # Calculate on GPU
        total_chars = int(char_gpu.size)
        
        # Transfer back for detailed analysis
        unique_chars_cpu = cp.asnumpy(unique_chars)
        counts_cpu = cp.asnumpy(counts)

        # Count types (done on CPU for char checking)
        char_counts = dict(zip(unique_chars_cpu, counts_cpu))
        letter_count = sum(count for byte, count in char_counts.items() 
                         if 65 <= byte <= 90 or 97 <= byte <= 122)
        digit_count = sum(count for byte, count in char_counts.items() 
                        if 48 <= byte <= 57)
        space_count = char_counts.get(32, 0)  # ASCII space
        
        cp.cuda.Stream.null.synchronize()
        
        end = time.time()
        elapsed = end - start

        print(f"\nResults:")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Unique bytes: {len(char_counts):,}")
        print(f"  Letters: {letter_count:,} ({100*letter_count/total_chars:.1f}%)")
        print(f"  Digits: {digit_count:,} ({100*digit_count/total_chars:.1f}%)")
        print(f"  Spaces: {space_count:,} ({100*space_count/total_chars:.1f}%)")
        
        print(f"\n GPU Time: {elapsed:.6f} seconds")
        
        return char_counts, elapsed
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("\n Database connection close")

def main():
    print("="*60)
    print("WIKIPEDIA TEXT ANALYSIS - CPU vs GPU Performance")
    print("="*60)
    
    # Configuration
    db_path = "/home/ansible/gpu-projects/shared-data/wiki/enwiki-20170820.db"
    article_limit = 1000000  # Start with subset for testing

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if len(sys.argv) > 2:
        article_limit = int(sys.argv[2])
    
    print(f"\nConfiguration:")
    print(f"  Database: {db_path}")
    print(f"  Article limit: {article_limit:,}")
    
    # Initialize analyzer
    analyzer = WikipediaAnalyzer(db_path)
    
    if not analyzer.connect():
        sys.exit(1)

    # Show database structure
    analyzer.get_table_info()
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    texts, query = analyzer.load_text_data(limit=article_limit)

    if not texts:
        print(" No text data found in database")
        analyzer.close()
        sys.exit()

    # Run benchmarks
    results = {}

    #1. Text length analysis
    cpu_lengths, cpu_time = analyzer.analyze_text_lengths_cpu(texts)
    results['lengths_cpu'] = cpu_time

    if GPU_AVAILABLE:
        gpu_lengths, gpu_time = analyzer.analyte_text_lengths_gpu(texts)
        results['lengths_gpu'] = gpu_time
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n Length Analysis Speedup: {speedup:.2f}x")

    # 2. Word frequency
    cpu_words, cpu_time = analyzer.word_frequency_cpu(texts, top_n=100)
    results['words_cpu'] = cpu_time
    
    # 3. Character distribution
    cpu_chars, cpu_time = analyzer.character_distribution_cpu(texts)
    results['chars_cpu'] = cpu_time
    
    if GPU_AVAILABLE:
        gpu_chars, gpu_time = analyzer.character_distribution_gpu(texts)
        results['chars_gpu'] = gpu_time
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n Character Analysis Speedup: {speedup:.2f}x")

    # Summary:
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    print("\nComputatipt Times:")
    for task, duration in results.items():
        print(f" {task:30s}: {duration:.6f} seconds")

    if GPU_AVAILABLE:
        print("\nSpeedup (CPU / GPU):")
        if 'lengths_cpu' in results and 'lengths_gpu' in results:
            speedup = results['lengths_cpu'] / results['lengths_gpu']
            print(f" Length Amalysis: {speedup:.2f}x")

        if 'chars_cpu' in results and 'chars_gpu' in results:
            speedup = results['chars_cpu'] / results['chars_gpu']
            print(f"  Character Distribution: {speedup:.2f}x")
    
    analyzer.close()

    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

if __name__ == "__main__":
    main()
        
                        























    