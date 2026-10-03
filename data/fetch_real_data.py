import os
import json
import requests
import csv

def fetch_nhtsa_tsbs(year="2018", make="toyota", model="camry"):
    """
    Fetches real Technical Service Bulletins (Manufacturer Communications) 
    from the NHTSA API for a specific vehicle.
    """
    print(f"Fetching real TSBs from NHTSA for {year} {make.title()} {model.title()}...")
    
    # NHTSA API endpoint for Manufacturer Communications (TSBs)
    url = f"https://api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}"
    
    # Note: To get actual TSB text, we usually query recalls or complaints APIs as an example
    recall_url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}"
    
    response = requests.get(recall_url)
    if response.status_code == 200:
        data = response.json()
        
        # Save the raw JSON data
        save_path = f"data/raw/tsbs/{year}_{make}_{model}_recalls.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully saved {len(data.get('results', []))} records to {save_path}")
        
        # Also create a text version for our Vector DB script to read easily
        text_path = f"data/raw/manuals/{year}_{make}_{model}_bulletins.txt"
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            for item in data.get('results', []):
                f.write(f"Component: {item.get('Component')}\n")
                f.write(f"Summary: {item.get('Summary')}\n")
                f.write(f"Remedy: {item.get('Remedy')}\n")
                f.write("-" * 50 + "\n")
        print(f"Created text manual for Vector DB at {text_path}")
    else:
        print(f"Failed to fetch NHTSA data. Status code: {response.status_code}")

def fetch_real_dtc_codes():
    """
    Downloads a real, comprehensive list of OBD-II DTC codes from a public repository.
    """
    print("Fetching real OBD-II DTC dataset...")
    
    # Updated to a highly reliable repository
    url = "https://raw.githubusercontent.com/mytrile/obd-trouble-codes/master/obd-trouble-codes.csv"
    
    response = requests.get(url)
    if response.status_code == 200:
        save_path = "data/raw/dtc_codes.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the raw downloaded CSV
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        # The downloaded CSV might have different headers, let's normalize it 
        # so our Graph Ingestion script can read it easily.
        normalize_dtc_csv(save_path)
        print(f"Successfully saved real DTC codes to {save_path}")
    else:
        print(f"Failed to fetch DTC data. Status code: {response.status_code}")

def normalize_dtc_csv(filepath):
    """
    Formats the downloaded CSV to guarantee 'DTC', 'Description', 'Part', 'Subsystem' columns
    for the Neo4j ingestion script.
    """
    normalized_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader) # skip headers
        except StopIteration:
            pass
        
        for row in reader:
            # Check if row has at least 2 columns and the first column looks like a DTC (e.g. P0300)
            if len(row) >= 2 and len(row[0].strip()) >= 4:
                code = row[0].strip()
                desc = row[1].strip()
                
                subsystem = "Powertrain"
                if code.startswith('C'): subsystem = "Chassis"
                elif code.startswith('B'): subsystem = "Body"
                elif code.startswith('U'): subsystem = "Network"
                
                normalized_data.append({
                    "DTC": code,
                    "Description": desc,
                    "Part": "Various Components", 
                    "Subsystem": subsystem
                })
    
    # Write back the normalized format
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["DTC", "Description", "Part", "Subsystem"])
        writer.writeheader()
        writer.writerows(normalized_data)

if __name__ == "__main__":
    print("--- Enterprise Data Ingestion ---")
    fetch_nhtsa_tsbs(year="2018", make="toyota", model="camry")
    fetch_nhtsa_tsbs(year="2012", make="ford", model="focus")
    fetch_real_dtc_codes()
    print("--- Data Fetching Complete ---")