#!/usr/bin/env python3

"""
Wikipedia SQLite Satabase Explorer
Analyze structure and content of enwiki-20170820.db
"""

import os
import sys
import sqlite3

if os.getuid() == 0:
    print("\n"+"-"*60 )
    print("ERROR: Don't run GPU as root! Use: su - ansible")
    print("-"*60 + "\n")
    sys.exit(1)

def explore_database(db_path):
    """Explore SQLite database structure and content"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("="*80)
        print("Database Structure Analysis")
        print("="*80)

        # Get all tables
        cursor.execute("Select name from sqlite_master Where type='table';")
        tables = cursor.fetchall()

        print(f"\nFound {len(tables)} table(s):")
        for table in tables:
            table_name = table[0]
            print(f"\n{'='*80}")
            print(f"TABLE: {table_name}")
            print('='*80)

            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            print("\nColumns:")
            for col in columns:
                col_id = col[0]
                name = col[1]
                type_ = col[2]
                notnull = col[3]
                default = col[4]
                pk = col[5]
            
                print(f" [{col_id}] {name:20s} {type_:15s} "
                      f"{'NOT NULL' if notnull else 'NULL':8s} "
                      f"{'PK' if pk else '':2s}")

            # Get row count
            cursor.execute(f"SELECT Count(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"\nTotal rows: {count:,}")

            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            samples = cursor.fetchall()

            if samples:
                print("\nSample data (first 3 rows):")
                col_names = [desc[0] for desc in cursor.description]

                for i, row in enumerate(samples, 1):
                    print(f"\n Row {i}:")
                    for col_name, value in zip(col_names, row):
                        # Truncate long test
                        if isinstance(value, str) and len(value) > 200:
                            display_value = value[:200] + "..."
                        else:
                            display_value = value
                        print(f"   {col_name}: {display_value}")

        conn.close

        print("\n" + "="*80)
        print("EXPLORATION COMPLETE")
        print("="*80)
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    db_path = "/home/ansible/gpu-projects/shared-data/wiki/enwiki-20170820.db"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print(f"Exploring database: {db_path}\n")
    explore_database(db_path)
                


























