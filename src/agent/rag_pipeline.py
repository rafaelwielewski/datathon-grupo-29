from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def build_knowledge_base_docs(cfg: dict) -> list[Document]:
    """Carrega e divide documentos do knowledge base em chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg['rag']['chunk_size'],
        chunk_overlap=cfg['rag']['chunk_overlap'],
    )
    all_docs: list[Document] = []
    for kb_dir in cfg['rag']['knowledge_base_dirs']:
        path = Path(kb_dir)
        if not path.exists():
            logger.warning('Knowledge base dir não encontrado: %s', kb_dir)
            continue
        loader = DirectoryLoader(
            str(path),
            glob='**/*.md',
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            show_progress=False,
        )
        docs = loader.load()
        all_docs.extend(splitter.split_documents(docs))
    logger.info('Knowledge base carregado: %d chunks', len(all_docs))
    return all_docs


def build_retriever(cfg: dict) -> BaseRetriever:
    """Constrói retriever FAISS com embeddings locais (sentence-transformers)."""
    docs = build_knowledge_base_docs(cfg)
    embeddings = HuggingFaceEmbeddings(model_name=cfg['rag']['embedding_model'])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={'k': cfg['rag']['k_results']})


def build_rag_tool(cfg: dict):
    """Retorna LangChain tool que faz busca semântica no knowledge base."""
    retriever = build_retriever(cfg)
    return create_retriever_tool(
        retriever,
        name='search_financial_knowledge',
        description=(
            'Searches the AAPL financial knowledge base. Use this tool to find context about: '
            'AAPL company profile, technical indicator interpretation, risk management guidelines, '
            'model limitations, and position sizing rules.'
        ),
    )
