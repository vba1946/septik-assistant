"""Индексация — единая коллекция septiki_knowledge с категориями."""
import os, re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_DIR = Path(__file__).parent / 'knowledge'
CHROMA_DIR = Path(__file__).parent / 'chromadb'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CATEGORY_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 9: 6}

def get_category(file_stem):
    match = re.match(r'^(\d+)', file_stem)
    return CATEGORY_MAP.get(int(match.group(1)), 0) if match else 0

def chunk_text(text, source):
    paragraphs = re.split(r'\n{2,}', text)
    chunks, current = [], []
    size = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if size + len(sent) > CHUNK_SIZE and current:
                    chunks.append({'text': '\n\n'.join(current), 'source': source})
                    current = [current[-1][-CHUNK_OVERLAP:]] if current[-1] else []
                    size = len(current[0]) if current else 0
                current.append(sent)
                size += len(sent)
            continue
        if size + len(para) > CHUNK_SIZE and current:
            chunks.append({'text': '\n\n'.join(current), 'source': source})
            current = [current[-1][-CHUNK_OVERLAP:]] if current[-1] else []
            size = len(current[0]) if current else 0
        current.append(para)
        size += len(para)
    if current:
        chunks.append({'text': '\n\n'.join(current), 'source': source})
    return chunks

def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('Укажите OPENAI_API_KEY')

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name='text-embedding-3-small'
    )
    try:
        client.delete_collection('septiki_knowledge')
    except Exception:
        pass

    collection = client.create_collection(
        name='septiki_knowledge', embedding_function=emb_fn,
        metadata={'hnsw:space': 'cosine'}
    )

    all_files = sorted(Path(KNOWLEDGE_DIR).iterdir())
    txt_files = [f for f in all_files if f.suffix == '.txt']
    all_chunks, all_ids, all_metas = [], [], []
    idx = 0

    for f in txt_files:
        cat = get_category(f.stem)
        text = f.read_text(encoding='utf-8', errors='replace')
        chunks = chunk_text(text, f.stem)
        for c in chunks:
            all_chunks.append(c['text'])
            all_ids.append(f'{f.stem}_{idx}')
            all_metas.append({'source': c['source'], 'category': cat})
            idx += 1

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(documents=all_chunks[i:batch_end], ids=all_ids[i:batch_end], metadatas=all_metas[i:batch_end])

    cats = set(m['category'] for m in all_metas)
    print(f'septiki_knowledge: {len(all_chunks)} чанков (категории: {cats})')

if __name__ == '__main__':
    main()
