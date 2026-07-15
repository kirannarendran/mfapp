import os
import re

def update_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Find the pattern for selected chips. E.g. in ComparisonView.jsx:
    # className={`px-2 py-0.5 text-[10px] rounded ${riskPeriod === '1Y' ? 'bg-sky-500 text-white' : 'text-slate-400 hover:text-slate-200'}`}
    
    # We will just replace 'bg-finance-primary text-white' with 'chip-selected'
    # and 'text-finance-muted hover:text-finance-heading' with 'chip-unselected'
    
    content = re.sub(r'bg-finance-primary text-white', 'chip-selected', content)
    content = re.sub(r'text-finance-muted hover:text-finance-heading', 'chip-unselected', content)
    
    # Also replace any text-finance-success/danger with explicit green/red logic if they were using old names
    # text-finance-danger -> text-[#DC2626] or text-finance-danger
    
    with open(path, 'w') as f:
        f.write(content)

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.jsx'):
            update_file(os.path.join(root, file))
