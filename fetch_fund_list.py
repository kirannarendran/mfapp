import requests
import csv

def fetch_all_funds_and_save_csv(filename="fund_list.csv"):
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch fund list")
        return
    
    funds = response.json()
    print(f"Fetched {len(funds)} funds")
    
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["schemeCode", "schemeName"])
        
        for fund in funds:
            scheme_code = fund.get("schemeCode")
            scheme_name = fund.get("schemeName")
            if scheme_code and scheme_name:
                writer.writerow([scheme_code, scheme_name])
    
    print(f"Saved fund list to {filename}")

if __name__ == "__main__":
    fetch_all_funds_and_save_csv()
