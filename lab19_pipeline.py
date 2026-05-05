import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from chromadb import Client
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-google-api-key-here")

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print("Vui lòng cung cấp Google API Key hợp lệ.")

def get_text_from_response(response):
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)

def extract_triples(text):
    prompt = """
    Bạn là một chuyên gia trích xuất thông tin.
    Đọc đoạn văn bản sau và trích xuất tất cả các quan hệ thực thể (Entity-Relation-Entity triples).
    Trả về ĐÚNG định dạng JSON sau, không kèm giải thích:
    {{"triples": [["Entity1", "RELATION", "Entity2"], ...]}}
    
    Đoạn văn bản:
    {text}
    """
    try:
        response = llm.invoke(prompt.format(text=text))
        content = get_text_from_response(response)
        # Regex to find json array
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
        data = json.loads(json_str)
        return data.get("triples", [])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error extracting triples: {e}")
        return []

def deduplicate_entity(entity):
    # Chuẩn hóa đơn giản: loại bỏ khoảng trắng thừa, viết thường (tuỳ chọn)
    return str(entity).strip()

def build_knowledge_graph_data(corpus_path):
    print("Đang đọc corpus và trích xuất Triples")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    all_triples = []
    
    # Track Token/Time (Mocking token count simply by word count for now, real usage should use callback)
    start_time = time.time()
    total_tokens_approx = 0
    
    for p in paragraphs:
        triples = extract_triples(p)
        total_tokens_approx += len(p.split()) * 1.5 # Thô
        for t in triples:
            if len(t) == 3:
                e1 = deduplicate_entity(t[0])
                rel = t[1].strip().upper()
                e2 = deduplicate_entity(t[2])
                all_triples.append((e1, rel, e2))
                
    extraction_time = time.time() - start_time
    print(f"Hoàn thành trích xuất {len(all_triples)} triples trong {extraction_time:.2f}s.")
    return all_triples, extraction_time, total_tokens_approx


def construct_graph(triples):
    print("Xây dựng đồ thị NetworkX")
    G = nx.DiGraph()
    for e1, rel, e2 in triples:
        G.add_edge(e1, e2, label=rel)
        
    # Trực quan hóa
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=10, font_weight='bold', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Tech Company Knowledge Graph")
    plt.savefig("knowledge_graph.png")
    print("Đã lưu đồ thị thành knowledge_graph.png")
    return G


def extract_entities_from_question(question):
    prompt = """
    Trích xuất danh sách CÁC thực thể chính từ câu hỏi sau. 
    Trả về danh sách các thực thể phân tách bằng dấu phẩy, không giải thích.
    Ví dụ: Apple, Microsoft
    Câu hỏi: {question}
    """
    response = llm.invoke(prompt.format(question=question))
    text = get_text_from_response(response).strip()
    return [e.strip() for e in text.split(",") if e.strip()]

def graph_search(G, start_node, hops=2):
    # Trùng khớp node (khử hoa/thường)
    start_node_lower = start_node.lower()
    matched_nodes = [n for n in G.nodes if n.lower() == start_node_lower or start_node_lower in n.lower()]
    
    if not matched_nodes:
        return []
    
    actual_start = matched_nodes[0]
    subgraph_edges = []
    
    # Lấy các cạnh trong vòng hops
    current_level = [actual_start]
    visited = set([actual_start])
    
    for _ in range(hops):
        next_level = []
        for node in current_level:
            for neighbor in G.successors(node):
                edge_data = G.get_edge_data(node, neighbor)
                subgraph_edges.append((node, edge_data['label'], neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)
            for neighbor in G.predecessors(node):
                edge_data = G.get_edge_data(neighbor, node)
                subgraph_edges.append((neighbor, edge_data['label'], node))
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)
        current_level = next_level
        
    return list(set(subgraph_edges))

def textualize_subgraph(edges):
    if not edges:
        return "Không có thông tin liên quan trong đồ thị."
    text = ". ".join([f"{u} {rel} {v}" for u, rel, v in edges]) + "."
    return text

def answer_graph_rag(G, question):
    start_time = time.time()
    
    entities = extract_entities_from_question(question)
    all_subgraph_edges = []
    for entity in entities:
        edges = graph_search(G, entity, hops=2)
        all_subgraph_edges.extend(edges)
        
    unique_edges = list(set(all_subgraph_edges))
    context = textualize_subgraph(unique_edges)
    
    prompt = f"""
    Dựa vào ngữ cảnh sau (được trích xuất từ Knowledge Graph), hãy trả lời câu hỏi.
    Nếu ngữ cảnh không có thông tin, hãy trả lời "Không đủ thông tin".
    Ngữ cảnh: {context}
    Câu hỏi: {question}
    """
    response = llm.invoke(prompt)
    
    query_time = time.time() - start_time
    return get_text_from_response(response), query_time


def setup_flat_rag(corpus_path):
    print(" Xây dựng Flat RAG (ChromaDB)")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chroma_client = Client(Settings(anonymized_telemetry=False))
    collection_name = "tech_corpus"
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass
        
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
    collection = chroma_client.create_collection(name=collection_name, embedding_function=google_ef)
    
    collection.add(
        documents=paragraphs,
        ids=[str(i) for i in range(len(paragraphs))]
    )
    return collection

def answer_flat_rag(collection, question):
    start_time = time.time()
    
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    context = " ".join(results['documents'][0])
    
    prompt = f"""
    Dựa vào ngữ cảnh sau, hãy trả lời câu hỏi.
    Ngữ cảnh: {context}
    Câu hỏi: {question}
    """
    response = llm.invoke(prompt)
    
    query_time = time.time() - start_time
    return get_text_from_response(response), query_time


questions = [
    # 1 hop
    "OpenAI được thành lập năm nào?",
    "Ai là CEO hiện tại của Google?",
    "Microsoft được sáng lập bởi ai?",
    "Meta trước đây tên là gì?",
    "Sản phẩm phần cứng nổi bật của Apple là gì?",
    "Ai là người sáng lập Amazon?",
    "Nvidia thiết kế loại chip nào quan trọng cho AI?",
    "DeepMind được mua lại bởi ai?",
    # 2 hop
    "Công ty do Mark Zuckerberg làm CEO đã phát triển mô hình ngôn ngữ mã nguồn mở nào?",
    "Công ty phát triển ChatGPT được thành lập bởi những ai?",
    "CEO của công ty phát triển Windows là ai?",
    "Công ty tạo ra AlphaGo được sáng lập vào năm nào?",
    # Tổng hợp / So sánh
    "Kể tên các công ty mà Elon Musk từng tham gia sáng lập hoặc mua lại.",
    "Ai đã sáng lập cả OpenAI và Anthropic?", # Trick question
    "Google và Meta có điểm chung gì về AI?",
    "Apple và Microsoft có CEO hiện tại là ai?",
    "Jensen Huang và Sundar Pichai là CEO của các công ty nào?",
    "Công ty nào phát triển Claude và ai sáng lập ra nó?",
    "Satya Nadella làm CEO của công ty nào và công ty đó đã đầu tư vào ai?",
    "Ai là người kế nhiệm Steve Jobs làm CEO của Apple?"
]

def run_benchmark(G, collection):
    print("--- Bắt đầu chạy Benchmark 20 câu hỏi ---")
    results = []
    
    for i, q in enumerate(questions):
        print(f"Processing Q{i+1}: {q}")
        flat_ans, flat_time = answer_flat_rag(collection, q)
        graph_ans, graph_time = answer_graph_rag(G, q)
        
        results.append({
            "Câu hỏi": q,
            "Flat RAG": flat_ans,
            "GraphRAG": graph_ans,
            "Flat Time (s)": round(flat_time, 2),
            "Graph Time (s)": round(graph_time, 2)
        })
        
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")
    print("Đã lưu kết quả so sánh vào benchmark_results.csv")
    return df

if __name__ == "__main__":
    corpus_path = "tech_corpus.txt"
    if not os.path.exists(corpus_path):
        print("Lỗi: Không tìm thấy tech_corpus.txt. Vui lòng tạo corpus trước.")
    else:
        # 1. Trích xuất
        triples, extract_time, extract_tokens = build_knowledge_graph_data(corpus_path)
        
        # 2. Xây dựng Graph
        G = construct_graph(triples)
        
        # 3. Setup Flat RAG
        collection = setup_flat_rag(corpus_path)
        
        # 4. Chạy Benchmark
        df_results = run_benchmark(G, collection)
        
        print("\nHOÀN THÀNH PIPELINE")
        print(f"Tổng thời gian trích xuất Graph: {extract_time:.2f} giây")
        print("Mở file benchmark_results.csv để xem kết quả chi tiết!")
