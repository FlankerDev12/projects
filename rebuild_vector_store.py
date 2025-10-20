"""
Quick script to rebuild vector store without prompts
"""
import os
import shutil
from RAG_engine import create_vector_store, PDF_FILES, VECTOR_STORE_DIR

print("=" * 70)
print("🔧 Rebuilding Vector Store")
print("=" * 70)
print()

# Remove old vector store if exists
if os.path.exists(VECTOR_STORE_DIR):
    print(f"🗑️  Removing old vector store: {VECTOR_STORE_DIR}")
    shutil.rmtree(VECTOR_STORE_DIR)
    print("✅ Old vector store removed")
    print()

print(f"📄 Processing {len(PDF_FILES)} PDF files...")
for i, pdf in enumerate(PDF_FILES, 1):
    size_mb = os.path.getsize(pdf) / (1024 * 1024)
    print(f"   {i}. {os.path.basename(pdf)} ({size_mb:.2f} MB)")
print()

print("🚀 Creating new vector store (this may take a few minutes)...")
print()

try:
    vector_store = create_vector_store(PDF_FILES)
    print()
    print("=" * 70)
    print("✅ SUCCESS! Vector store created!")
    print("=" * 70)
    print()
    print(f"📁 Location: {os.path.abspath(VECTOR_STORE_DIR)}")
    print()
    print("✨ Your RAG engine is ready to use!")
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ ERROR")
    print("=" * 70)
    print(f"{str(e)}")
    print()
    print("💡 Tip: Check your internet connection if downloading models failed")
