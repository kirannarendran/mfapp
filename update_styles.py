import os
import re

def update_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # 1. Remove all gradients and drop shadows
    content = re.sub(r'bg-gradient-to-[a-z]+ ', '', content)
    content = re.sub(r'from-[a-z]+-[0-9]+ ', '', content)
    content = re.sub(r'to-[a-z]+-[0-9]+ ', '', content)
    content = re.sub(r'bg-clip-text ', '', content)
    content = re.sub(r'text-transparent ', '', content)
    content = re.sub(r'shadow-[a-z]+-[0-9]+/[0-9]+ ', '', content)

    # 2. Update specific elements according to instructions
    # Headers (App.jsx)
    content = content.replace('bg-white border-b border-slate-200 sticky', 'bg-finance-surface border-b border-finance-border sticky shadow-sm')
    content = content.replace('bg-white border border-slate-200', 'bg-finance-surface border border-finance-border')
    
    # Text colors
    content = re.sub(r'text-slate-900', 'text-finance-heading', content)
    content = re.sub(r'text-slate-800', 'text-finance-heading', content)
    content = re.sub(r'text-slate-600', 'text-finance-text', content)
    content = re.sub(r'text-slate-500', 'text-finance-muted', content)
    
    # Primary colors (blue)
    content = re.sub(r'text-sky-600', 'text-finance-primary', content)
    content = re.sub(r'hover:text-sky-700', 'hover:text-finance-primary-dark', content)
    content = re.sub(r'bg-sky-500', 'bg-finance-primary', content)
    content = re.sub(r'hover:bg-sky-600', 'hover:bg-finance-primary-dark', content)
    content = re.sub(r'border-sky-500', 'border-finance-primary', content)
    content = re.sub(r'focus:border-sky-500', 'focus:border-finance-primary', content)
    content = re.sub(r'focus:ring-sky-500/20', 'focus:ring-finance-primary/20', content)
    
    # Danger/Success/Warning
    content = re.sub(r'text-emerald-600', 'text-finance-success', content)
    content = re.sub(r'text-green-600', 'text-finance-success', content)
    content = re.sub(r'text-red-500', 'text-finance-danger', content)
    content = re.sub(r'text-red-600', 'text-finance-danger', content)
    content = re.sub(r'text-yellow-500', 'text-finance-warning', content)
    content = re.sub(r'text-amber-500', 'text-finance-warning', content)
    
    # Backgrounds and borders
    content = re.sub(r'bg-slate-50', 'bg-finance-bg', content)
    content = re.sub(r'bg-slate-100', 'bg-finance-bg', content)
    content = re.sub(r'bg-slate-200', 'bg-finance-border', content)
    content = re.sub(r'bg-slate-300', 'bg-finance-border', content)
    
    content = re.sub(r'border-slate-100', 'border-finance-border', content)
    content = re.sub(r'border-slate-200', 'border-finance-border', content)
    content = re.sub(r'border-slate-300', 'border-finance-border', content)
    
    # Table styling specific changes
    # "Use dark navy with white text for table headers" -> bg-finance-table-header text-white
    content = re.sub(r'thead className=".*?"', 'thead className="bg-finance-table-header text-white text-sm uppercase tracking-wider"', content)
    content = re.sub(r'th className=".*?"', 'th className="px-6 py-4 font-medium"', content)
    # Extremely light row highlighting
    content = re.sub(r'hover:bg-finance-bg', 'hover:bg-blue-50/50', content)
    content = re.sub(r'bg-white', 'bg-finance-surface', content)

    with open(path, 'w') as f:
        f.write(content)

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.jsx'):
            update_file(os.path.join(root, file))
