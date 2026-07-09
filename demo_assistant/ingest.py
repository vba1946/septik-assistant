"""Индексация базы знаний — создаёт две коллекции: septiki_pro и septiki_simple."""
import os, re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_DIR = Path(__file__).parent / 'knowledge'
CHROMA_DIR = Path(__file__).parent / 'chromadb'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text: str, source: str) -> list[dict]:
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
                    overlap_text = '\n\n'.join(current)[-CHUNK_OVERLAP:]
                    current = [overlap_text] if overlap_text else []
                    size = len(overlap_text)
                current.append(sent)
                size += len(sent)
            continue
        if size + len(para) > CHUNK_SIZE and current:
            chunks.append({'text': '\n\n'.join(current), 'source': source})
            overlap_text = '\n\n'.join(current)[-CHUNK_OVERLAP:]
            current = [overlap_text] if overlap_text else []
            size = len(overlap_text)
        current.append(para)
        size += len(para)
    if current:
        chunks.append({'text': '\n\n'.join(current), 'source': source})
    return chunks

def index_collection(collection_name: str, files: list):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('Укажите OPENAI_API_KEY')

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name='text-embedding-3-small'
    )

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={'hnsw:space': 'cosine'}
    )

    all_chunks, all_ids, all_metas = [], [], []
    idx = 0
    for f in files:
        text = f.read_text(encoding='utf-8', errors='replace')
        chunks = chunk_text(text, f.stem)
        for c in chunks:
            all_chunks.append(c['text'])
            all_ids.append(f'{f.stem}_{idx}')
            all_metas.append({'source': c['source']})
            idx += 1

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            ids=all_ids[i:batch_end],
            metadatas=all_metas[i:batch_end]
        )
    print(f'  {collection_name}: {len(all_chunks)} чанков')
    return len(all_chunks)

def main():
    all_files = sorted(KNOWLEDGE_DIR.iterdir())
    txt_files = [f for f in all_files if f.suffix == '.txt']

    pro_files = [f for f in txt_files if 'PRO' in f.stem]
    simple_files = [f for f in txt_files if 'PRO' not in f.stem]

    print(f'Файлов: PRO={len(pro_files)}, Simple={len(simple_files)}')

    print('Индексация septiki_pro...')
    n_pro = index_collection('septiki_pro', pro_files)

    print('Индексация septiki_simple...')
    n_simple = index_collection('septiki_simple', simple_files)

    print(f'\nГотово: PRO={n_pro} чанков, Simple={n_simple} чанков')
    print(f'Хранилище: {CHROMA_DIR}')

if __name__ == '__main__':
    main()
