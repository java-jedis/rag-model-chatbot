import os
import json
from pprint import pprint

def explore_json_files(directory):
    """Explore JSON files in the given directory and print their structure."""
    json_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    if json_files:
        # Print structure of the first file
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            print("\nSample JSON structure:")
            pprint(data if len(str(data)) < 1000 else 
                  {k: type(v).__name__ for k, v in data.items()})

if __name__ == "__main__":
    for volume in ["volume_i", "volume_ii"]:
        print(f"\nExploring {volume}:")
        explore_json_files(volume)
