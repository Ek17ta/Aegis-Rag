import os

fixes = {
    "retrieval.py": [
        ("from langchain.schema import Document", "from langchain_core.documents import Document"),
        ("from langchain.prompts import ChatPromptTemplate", "from langchain_core.prompts import ChatPromptTemplate"),
        ("from langchain_community.vectorstores import Chroma", "from langchain_community.vectorstores import Chroma"),
    ],
    "ingestion.py": [
        ("from langchain.schema import Document", "from langchain_core.documents import Document"),
        ("from langchain_community.vectorstores import Chroma", "from langchain_community.vectorstores import Chroma"),
    ],
}

for filename, replacements in fixes.items():
    if not os.path.exists(filename):
        print(f"Skipping {filename} - not found")
        continue
    with open(filename, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filename, "w") as f:
        f.write(content)
    print(f"✅ {filename} fixed!")

print("\nAll done! Now run: python -m streamlit run app.py")