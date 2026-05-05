# BÁO CÁO LAB DAY 19 - GRAPHRAG VS FLAT RAG

## PHẦN 1 – NGHIÊN CỨU (Lý thuyết)

### 1. Entity Extraction: Cách LLM phân biệt thực thể (Node) với thuộc tính (Attribute)
LLM phân biệt thực thể và thuộc tính chủ yếu thông qua ngữ nghĩa của từ vựng và cấu trúc ngữ pháp (ví dụ: danh từ riêng thường là thực thể, tính từ/phó từ hoặc danh từ chung thường là thuộc tính). Tuy nhiên, để hệ thống hoạt động chính xác trong việc xây dựng Knowledge Graph, chúng ta sử dụng **Prompt Engineering**.
Bằng cách cung cấp cấu trúc rõ ràng trong prompt, ví dụ yêu cầu trả về định dạng `{"triples": [["Entity1", "RELATION", "Entity2"]]}` hoặc giải thích rõ "chỉ lấy các đối tượng độc lập có thể làm Node (ví dụ tên người, tên công ty)", LLM bị ép buộc đưa các thuộc tính (như tuổi, màu sắc, ngày tháng) thành một RELATION kết nối đến một Node giá trị (ví dụ `(Sam Altman, AGE, 38)`) thay vì nhầm lẫn nó với một thực thể trung tâm.

### 2. Graph Construction: Tại sao khử trùng lặp (Deduplication) lại quan trọng?
Trong quá trình trích xuất bằng LLM từ nhiều đoạn văn khác nhau, một thực thể duy nhất có thể được biểu diễn bằng nhiều cách khác nhau, ví dụ: "Sam Altman", "Sam Altman (CEO)", "Altman", hoặc "OpenAI" vs "Open AI".
Nếu không có quá trình khử trùng lặp (Entity Resolution/Deduplication), đồ thị sẽ tạo ra nhiều Node rời rạc cho cùng một đối tượng. Điều này dẫn đến sự phân mảnh thông tin, làm đứt gãy đồ thị và khiến quá trình truy vấn (như duyệt BFS) không thể đi qua các Node đó để thu thập đủ ngữ cảnh. Khử trùng lặp giúp hợp nhất chúng thành một Node duy nhất, tăng mật độ liên kết và tính toàn vẹn của Knowledge Graph.

### 3. Querying: BFS vs Vector Search
- **Vector Search (Flat RAG)**: Hoạt động dựa trên không gian nhúng (embedding space). Hệ thống tìm kiếm các đoạn văn bản có ý nghĩa tương đồng nhất với câu hỏi. 
  - *Ưu điểm*: Nhanh, dễ setup, nắm bắt tốt ngữ nghĩa của câu hỏi mà không cần khớp từ khóa chính xác.
  - *Nhược điểm*: Rất kém trong các câu hỏi đa luồng (multi-hop) vì thông tin có thể nằm rải rác ở nhiều đoạn văn không liên quan về mặt ngữ nghĩa với nhau nhưng lại liên quan qua chuỗi quan hệ.
- **Duyệt đồ thị theo chiều rộng BFS (GraphRAG)**: Bắt đầu từ một hoặc nhiều node trung tâm (Entity được trích xuất từ câu hỏi) và lan truyền ra xung quanh theo số bước (hops) xác định.
  - *Ưu điểm*: Cực kỳ mạnh mẽ trong việc giải quyết câu hỏi "multi-hop" (ví dụ: A liên quan đến B, B liên quan đến C -> hỏi sự liên quan giữa A và C). Đặc biệt hiệu quả khi so sánh hoặc đối chiếu nhiều chủ thể nếu trích xuất đủ các thực thể từ câu hỏi.
  - *Nhược điểm*: Phụ thuộc vào chất lượng của Knowledge Graph (nếu trích xuất thiếu quan hệ, BFS sẽ không tìm thấy).

---

## PHẦN 2 – BẢNG SO SÁNH BENCHMARK (KẾT QUẢ THỰC TẾ ĐÃ CẬP NHẬT)

Dựa trên bản cập nhật tối ưu hóa thuật toán truy vấn nhiều thực thể (Multi-Entity Query) trong `lab19_pipeline.py`, dưới đây là kết quả benchmark:

| # | Câu hỏi | Flat RAG Answer | GraphRAG Answer | Đúng? (Flat/Graph) | Ghi chú |
|---|---------|----------------|----------------|---------------------|---------|
| 1 | OpenAI được thành lập năm nào? | 2015 | 2015 | ✅ / ✅ | Câu hỏi 1-hop, cả hai đều trả lời tốt. |
| 2 | CEO của công ty phát triển Windows là ai? | Satya Nadella | Satya Nadella | ✅ / ✅ | Câu hỏi 2-hop (Windows -> Microsoft -> Satya). Cả hai đều làm tốt. |
| 3 | Satya Nadella làm CEO của công ty nào và công ty đó đã đầu tư vào ai? | Microsoft, đầu tư vào OpenAI | Microsoft, đối tác chiến lược OpenAI | ✅ / ✅ | Câu hỏi Multi-hop phức tạp. Cả 2 đều xử lý tốt. (GraphRAG còn nhận diện rõ ràng "partnership" thay vì "invest"). |
| 4 | Apple và Microsoft có CEO hiện tại là ai? | Tim Cook và Satya Nadella | Tim Cook và Satya Nadella | ✅ / ✅ | **Đã khắc phục hoàn toàn!** Bằng cách trích xuất cả "Apple" và "Microsoft" từ câu hỏi và gộp ngữ cảnh, GraphRAG trả lời vô cùng chính xác. |
| 5 | Meta trước đây tên là gì? | Facebook | Không đủ thông tin. | ✅ / ❌ | Có thể trong Knowledge Graph, LLM trích xuất chưa bắt được mối quan hệ "đổi tên" (RENAMED) giữa Meta và Facebook. Đây là rủi ro của GraphRAG khi phụ thuộc vào chất lượng trích xuất ban đầu. |
| 6 | Google và Meta có điểm chung gì về AI? | Cả hai phát triển mô hình ngôn ngữ... | Cả hai đều phát triển các mô hình AI... | ✅ / ✅ | **Đã khắc phục hoàn toàn!** GraphRAG giờ đây thu thập thông tin về cả Google và Meta, từ đó đưa ra câu trả lời so sánh hoàn chỉnh y như Flat RAG. |

**Phân tích kết quả (Sau khi nâng cấp Multi-Entity):**
- **GraphRAG** đã thể hiện đúng sức mạnh thực sự của nó đối với các câu hỏi đa chủ thể (multi-subject) như Q4 và Q6. Khả năng duyệt tìm Subgraph cho nhiều thực thể và hợp nhất chúng (Union) cung cấp một Context có tính logic cao, loại bỏ hoàn toàn các thông tin nhiễu mà Flat RAG có thể mắc phải nếu Corpus quá lớn.
- **Điểm yếu duy nhất còn lại của GraphRAG**: Nếu bước trích xuất ban đầu (Extract Triples) vô tình bỏ sót một mệnh đề quan trọng (Ví dụ: sự liên quan giữa Meta và Facebook ở Q5), thì cho dù thuật toán BFS có giỏi đến đâu cũng không thể tìm ra thông tin đó. Đây là đặc điểm "Garbage In, Garbage Out" của GraphRAG.

---

## PHẦN 3 – PHÂN TÍCH CHI PHÍ & THỜI GIAN

Dựa trên kết quả chạy thực tế của pipeline:

1. **Token Usage (Indexing - Trích xuất Triple)**: 
   - Quá trình trích xuất đã tạo ra thành công **106 triples** từ 15 đoạn văn của corpus.
   - Do sử dụng model `gemini-flash-latest`, chi phí indexing cực kỳ rẻ và hiệu quả.

2. **Thời gian xử lý (Indexing Time)**:
   - **GraphRAG**: Mất **~64 giây** để LLM đọc và trích xuất toàn bộ triples cho corpus. Thời gian này khá đáng kể do phải thực hiện nhiều API calls qua mạng cho từng đoạn văn.
   - **Flat RAG**: Sử dụng `GoogleGenerativeAiEmbeddingFunction` thông qua API, việc tạo bộ sưu tập và đẩy Embedding lên ChromaDB chỉ mất khoảng **vài giây**.
   - *Kết luận*: Flat RAG indexing nhanh hơn rất nhiều. GraphRAG đòi hỏi chi phí tính toán (Token + Thời gian) lớn ở bước xây dựng Knowledge Graph.

3. **Query Time (Thời gian truy vấn)**:
   - **Flat RAG Time**: Trung bình từ **2.5s đến 5.5s** cho mỗi câu hỏi.
   - **GraphRAG Time**: Trung bình từ **4.5s đến 10.0s** (có câu lên tới 14s) cho mỗi câu hỏi.
   - *Kết luận*: GraphRAG tốn nhiều thời gian hơn trong lúc truy vấn do phải gọi LLM 2 lần (1 lần trích xuất danh sách Entity gốc, 1 lần gọi để sinh câu trả lời dựa trên ngữ cảnh Subgraph tổng hợp).

---
**File liên quan:**
- Mã nguồn chạy pipeline: `lab19_pipeline.py`
- Dữ liệu thực thể: `tech_corpus.txt`
- Bảng Benchmark CSV chi tiết: `benchmark_results.csv`
- Trực quan hóa Graph: `knowledge_graph.png`
