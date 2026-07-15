import os
import re

mapping = {
    r'bg-slate-900': 'bg-white',
    r'bg-slate-800/50': 'bg-slate-50',
    r'bg-slate-800': 'bg-slate-50',
    r'bg-slate-700/50': 'bg-slate-100',
    r'bg-slate-700': 'bg-slate-200',
    r'bg-slate-600': 'bg-slate-300',
    
    r'border-slate-800': 'border-slate-200',
    r'border-slate-700': 'border-slate-300',
    
    r'text-slate-200': 'text-slate-900',
    r'text-slate-300': 'text-slate-800',
    r'text-slate-400': 'text-slate-600',
    
    r'text-sky-400': 'text-sky-600',
    r'hover:text-sky-300': 'hover:text-sky-700',
    
    r'text-emerald-400': 'text-emerald-600',
    r'text-green-400': 'text-emerald-600',
    
    r'from-sky-400': 'from-sky-600',
    r'to-emerald-400': 'to-emerald-600',
}

def convert_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    for old, new in mapping.items():
        # Match class boundaries so we don't accidentally replace partial strings
        content = re.sub(r'(?<=[\s"\'`])' + old + r'(?=[\s"\'`])', new, content)
    
    with open(path, 'w') as f:
        f.write(content)

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.jsx'):
            convert_file(os.path.join(root, file))

# Update index.css custom variables
css_path = 'src/index.css'
with open(css_path, 'r') as f:
    css = f.read()
    
css = css.replace('--bg-color: #0f172a;', '--bg-color: #ffffff;')
css = css.replace('--card-bg: #1e293b;', '--card-bg: #f8fafc;')
css = css.replace('--text-primary: #f8fafc;', '--text-primary: #0f172a;')
css = css.replace('--text-secondary: #94a3b8;', '--text-secondary: #475569;')
css = css.replace('--border-color: #334155;', '--border-color: #e2e8f0;')

with open(css_path, 'w') as f:
    f.write(css)

print("Conversion complete.")
