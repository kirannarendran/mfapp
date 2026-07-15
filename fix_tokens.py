import os
import re

def update_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Replace words safely
    # text-finance-heading -> text-finance-text-primary
    content = re.sub(r'\btext-finance-heading\b', 'text-finance-text-primary', content)
    # text-finance-muted -> text-finance-text-secondary
    content = re.sub(r'\btext-finance-muted\b', 'text-finance-text-secondary', content)
    # text-finance-text (only if not already followed by -primary or -secondary)
    content = re.sub(r'\btext-finance-text(?!-(primary|secondary))\b', 'text-finance-text-primary', content)
    
    with open(path, 'w') as f:
        f.write(content)

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.jsx'):
            update_file(os.path.join(root, file))
