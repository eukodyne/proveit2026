#!/usr/bin/env python3
"""Check what's stored in Milvus vector database."""

import os
from pymilvus import connections, utility, Collection

# Connect to Milvus (use 'milvus' host in Docker, 'localhost' otherwise)
host = "milvus" if os.path.exists("/.dockerenv") else "localhost"
print(f"Connecting to Milvus at {host}:19530...")
connections.connect(host=host, port="19530")

# List all collections
collections = utility.list_collections()
print(f"\nCollections found: {len(collections)}")

if not collections:
    print("No collections in Milvus.")
else:
    for name in collections:
        print(f"\n{'='*50}")
        print(f"Collection: {name}")
        print('='*50)

        col = Collection(name)
        col.load()

        # Get schema info
        print(f"Description: {col.description}")
        print(f"Num entities: {col.num_entities}")

        print("\nSchema fields:")
        for field in col.schema.fields:
            print(f"  - {field.name}: {field.dtype.name}", end="")
            if hasattr(field, 'dim') and field.dim:
                print(f" (dim={field.dim})", end="")
            if field.is_primary:
                print(" [PRIMARY]", end="")
            print()

        # Show sample data if exists
        if col.num_entities > 0:
            print(f"\nSample records (first 3):")
            results = col.query(
                expr="",
                output_fields=["*"],
                limit=3
            )
            for i, r in enumerate(results):
                print(f"\n  Record {i+1}:")
                for k, v in r.items():
                    if k == "embedding":
                        print(f"    {k}: [{v[0]:.4f}, {v[1]:.4f}, ... ] (len={len(v)})")
                    elif isinstance(v, str) and len(v) > 100:
                        print(f"    {k}: {v[:100]}...")
                    else:
                        print(f"    {k}: {v}")

connections.disconnect("default")
print("\nDone.")
