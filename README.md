# GraphRAG vs Flat RAG Pipeline

Dự án này là một bài thực hành (Lab) nhằm xây dựng, thử nghiệm và so sánh trực tiếp hiệu năng giữa hai phương pháp RAG (Retrieval-Augmented Generation) phổ biến hiện nay:
1. **Flat RAG**: Tìm kiếm theo Vector (Vector Search) sử dụng ChromaDB.
2. **GraphRAG**: Trích xuất thực thể (Triples) và tìm kiếm thông qua Duyệt đồ thị (BFS) sử dụng NetworkX.

## Cấu trúc Hệ thống và Mô hình sử dụng

Dự án sử dụng bộ công cụ **Langchain** và **Google Generative AI** (Gemini) để xử lý dữ liệu:

*   **LLM Model**: `gemini-flash-latest` (được sử dụng cho cả việc trích xuất Knowledge Graph Triples, trích xuất Entity từ câu hỏi, và tạo câu trả lời cuối cùng).
*   **Embedding Model**: `GoogleGenerativeAiEmbeddingFunction` (được sử dụng để nhúng các đoạn văn bản trong `tech_corpus.txt` vào ChromaDB).
*   **Vector Database**: `ChromaDB` (lưu trữ cục bộ).
*   **Graph Database**: `NetworkX` (được dùng để lưu trữ và duyệt đồ thị trên RAM).

## Cấu trúc thư mục

*   `lab19_pipeline.py`: Mã nguồn chính chứa toàn bộ pipeline từ xây dựng Graph, setup Flat RAG đến chạy Benchmark.
*   `tech_corpus.txt`: Dữ liệu đầu vào (Corpus) chứa thông tin về các công ty công nghệ.
*   `requirements.txt`: Danh sách các thư viện Python cần thiết.
*   `.env.example`: File mẫu chứa cấu hình biến môi trường.
*   `Report_Lab19.md`: Báo cáo phân tích chuyên sâu về lý thuyết và kết quả so sánh.

## Hướng dẫn cài đặt và chạy thử

### Bước 1: Cài đặt thư viện
Yêu cầu Python 3.9+ (Dự án đang chạy trên Python 3.14). Mở terminal và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 2: Thiết lập API Key
Dự án sử dụng Google Gemini API. Bạn cần tạo một file `.env` tại thư mục gốc (hoặc copy từ `.env.example`) và điền API Key của bạn:
```env
GOOGLE_API_KEY=your-google-api-key-here
```

### Bước 3: Chạy Pipeline
Thực thi file Python chính để hệ thống bắt đầu quá trình trích xuất và trả lời 20 câu hỏi Benchmark:
```bash
python lab19_pipeline.py
```
*(Nếu bạn sử dụng môi trường Python cụ thể như trong Lab này, lệnh sẽ là: `D:\Python314\python.exe lab19_pipeline.py`)*

### Bước 4: Xem kết quả
Sau khi script chạy hoàn tất, hệ thống sẽ tự động tạo ra 2 file output:
1.  `knowledge_graph.png`: Hình ảnh trực quan hóa sơ đồ Knowledge Graph mạng lưới các công ty công nghệ.
2.  `benchmark_results.csv`: File Excel/CSV chứa kết quả câu trả lời song song giữa Flat RAG và GraphRAG cùng với thời gian thực thi của từng phương pháp.

Đọc thêm file `Report_Lab19.md` để xem phân tích chi tiết tại sao GraphRAG lại vượt trội hơn ở các câu hỏi đa chủ thể (Multi-Subject) và thời gian thực thi của hai hệ thống.
