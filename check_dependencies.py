"""
Dependency Checker - Verifica quali file usano retrieval/retriever
"""
import os
import re
from pathlib import Path

def check_dependencies(root_dir):
    """Scansiona tutti i file Python per trovare import di retriever"""
    
    patterns = {
        'retrieval': re.compile(r'from\s+.*retrieval\s+import|import\s+.*retrieval', re.IGNORECASE),
        'retriever': re.compile(r'from\s+.*retriever\s+import|import\s+.*retriever(?!_fixed)', re.IGNORECASE),
        'retriever_fixed': re.compile(r'from\s+.*retriever_fixed\s+import|import\s+.*retriever_fixed', re.IGNORECASE),
    }
    
    results = {
        'retrieval': [],
        'retriever': [],
        'retriever_fixed': []
    }
    
    # Scansiona tutti i file .py
    for path in Path(root_dir).rglob('*.py'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for module_name, pattern in patterns.items():
                if pattern.search(content):
                    rel_path = path.relative_to(root_dir)
                    results[module_name].append(str(rel_path))
        except Exception as e:
            print(f"⚠️  Errore leggendo {path}: {e}")
    
    return results

def print_report(results):
    """Stampa report delle dipendenze"""
    print("\n" + "="*70)
    print("📊 DEPENDENCY ANALYSIS REPORT")
    print("="*70)
    
    for module_name, files in results.items():
        print(f"\n🔍 Files that import '{module_name}':")
        if files:
            for f in files:
                print(f"   ✓ {f}")
        else:
            print(f"   ✅ NESSUNO (safe to remove)")
    
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    # Safe to remove?
    if not results['retriever']:
        print("✅ retriever.py is SAFE to remove (not used)")
    else:
        print("⚠️  retriever.py is USED by:")
        for f in results['retriever']:
            print(f"   - {f}")
    
    if not results['retriever_fixed']:
        print("✅ retriever_fixed.py is SAFE to remove (not used)")
    else:
        print("⚠️  retriever_fixed.py is USED by:")
        for f in results['retriever_fixed']:
            print(f"   - {f}")
    
    if results['retrieval']:
        print(f"⚠️  retrieval.py is REQUIRED by {len(results['retrieval'])} file(s):")
        for f in results['retrieval']:
            print(f"   - {f}")
    
    print("\n" + "="*70)

def check_runtime_usage():
    """Verifica anche menzioni non-import"""
    print("\n🔎 Checking for runtime references (RetrievalService, Retriever)...")
    
    root = Path.cwd()
    
    classes_to_check = {
        'RetrievalService': [],
        'Retriever': []
    }
    
    for path in root.rglob('*.py'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for class_name in classes_to_check.keys():
                if class_name in content:
                    rel_path = path.relative_to(root)
                    classes_to_check[class_name].append(str(rel_path))
        except:
            pass
    
    print("\n📌 Runtime class usage:")
    for class_name, files in classes_to_check.items():
        if files:
            print(f"\n   {class_name} found in:")
            for f in set(files):
                print(f"      - {f}")

if __name__ == "__main__":
    print("🚀 Starting dependency analysis...")
    print(f"📁 Root directory: {Path.cwd()}")
    
    results = check_dependencies(Path.cwd())
    print_report(results)
    check_runtime_usage()
    
    print("\n✅ Analysis complete!")