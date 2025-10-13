import json
import os
from tqdm import tqdm

# Path configuration
BASE_DIR = "./Datasets/CATH4.2"
TXT_DIR = "output_groups"
OUTPUT_DIR = "Seq_split_json"
CHAIN_SET_FILE = os.path.join(BASE_DIR, "chain_set.jsonl")

def create_output_dir():
    """Create output directory if it does not exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def filter_chains_by_txt(txt_file_path):
    """
    Process a single TXT file and generate the corresponding JSON file
    containing matched chain entries from chain_set.jsonl
    """
    txt_filename = os.path.basename(txt_file_path)
    json_filename = txt_filename.replace(".txt", ".json")
    output_path = os.path.join(OUTPUT_DIR, json_filename)
    
    # Read chain names from TXT file
    with open(txt_file_path, 'r') as f:
        chain_names = [line.strip().split()[0] for line in f if line.strip()]
    
    chain_set = set(chain_names)
    print(f"Processing file: {txt_filename}, contains {len(chain_set)} chains")
    
    matched_entries = []
    
    # Match chains from chain_set.jsonl
    with open(CHAIN_SET_FILE, 'r') as f:
        for line in tqdm(f, desc=f"Scanning {txt_filename}"):
            try:
                entry = json.loads(line)
                if entry['name'] in chain_set:
                    for key in entry['coords']:
                        # Keep as list for JSON serialization
                        pass
                    matched_entries.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line: {line[:50]}...")
    
    # Save matched entries to JSON file
    with open(output_path, 'w') as f_out:
        json.dump(matched_entries, f_out, indent=2)
    
    print(f"Saved: {output_path}, contains {len(matched_entries)} entries")
    return len(matched_entries)

def process_all_txt_files():
    """Process all similarity_*.txt files in the TXT_DIR"""
    create_output_dir()
    
    txt_files = [f for f in os.listdir(TXT_DIR) 
                if f.startswith("similarity_") and f.endswith(".txt")]
    
    print(f"Found {len(txt_files)} txt files to process")
    
    for txt_file in sorted(txt_files):
        txt_path = os.path.join(TXT_DIR, txt_file)
        filter_chains_by_txt(txt_path)

if __name__ == "__main__":
    process_all_txt_files()
    print("All files processed successfully!")
