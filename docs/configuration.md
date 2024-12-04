# RHEL AI Pro Suite Configuration
RHEL AI Pro Suite에서 사용하는 Configuration에 대해 설명합니다.

## configuration.yml 샘플

```yaml
logging:
  # Setting the log level: CRITICAL, ERROR, WARNING, INFO, DEBUG
  level: 'DEBUG'

plugin_dir: ./plugins

http:
  host: 0.0.0.0
  port: 8888

rag:
  strict_answer: true
  source_visible: true
  context_visible: true
  search_size: 3
  context_size: 2
  min_similarity: 0.01
  retrieval_type: hybrid  # lexical, semantic, hybrid

demo:
  chat_playground:
    enabled: true
    host: localhost
    port: 8503
  evaluation_dashboard:
    enabled: true
    host: localhost
    port: 8504

plugins:
  chunker:
    recursive-character:
      chunk_size: 512
      chunk_overlap: 60
      tiktoken_encoding_name: cl100k_base

  filter:
    document:
      - path: context.metadata._source.metadata.status
        term: match  # not_match, greater_than, less_than
        value:
          - trained

  embedding_model:
    huggingface-embedding:
      model_path: intfloat/multilingual-e5-large-instruct
      device: cpu

  vector_store:
    elasticsearch:
      hosts: [ "http://127.0.0.1:9200" ]
      index_name: ai_odyssey_demo_documents-000001
      embedding_model: huggingface-embedding

  reranker:
    bge-reranker:
      model_path: BAAI/bge-reranker-v2-m3

  llm:
    remote:
      host: http://127.0.0.1:8000
      model: merlinite-7b-lab-Q4_K_M.gguf

  evaluation:
    rhel-ai:
      llm: remote
      critic_llm:
        host: http://127.0.0.1:8000
        model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
      testset_index_name: ai_odyssey_demo_testset-000001
      evaluation_index_name: ai_odyssey_demo_evaluation-000001
      metrics:
        - faithfulness
        - answer_relevancy
        - context_precision
        - context_recall
        - context_entity_recall
        - noise_sensitivity_relevant
        - answer_similarity
```

## 로깅 설정

```yaml
logging:
  # Setting the log level: CRITICAL, ERROR, WARNING, INFO, DEBUG
  level: 'DEBUG'
```
- `level`: 로그의 Level을 설정 (로그 Level: CRITICAL, ERROR, WARNING, INFO, DEBUG). 기본값은 `DEBUG`

## 서버 설정

```yaml
plugin_dir: ./plugins

http:
  host: 0.0.0.0
  port: 8888

rag:
  strict_answer: true
  source_visible: true
  context_visible: true
  search_size: 3
  context_size: 2
  min_similarity: 0.01
  retrieval_type: hybrid    # lexical, semantic, hybrid

demo:
  chat_playground:
    enabled: true
    host: localhost
    port: 8503
  evaluation_dashboard:
    enabled: true
    host: localhost
    port: 8504
```

- `plugin_dir`: 애플리케이션 내 plugin들이 위치한 최상위 디렉토리. 기본값은 `./plugins`

- `http`
    - `host`: 서버의 IP 주소 또는 도메인명
    - `port`: 서버 사용 포트

- `rag`: RAG(Retrieval-Augmented Generation) 관련 설정
    - `strict_answer`: 검색된 문서가 존재하지 않을 시, 사전 정의된 답변을 클라이언트에게 전달할지 여부. 기본값은 `False`

> [!NOTE]
> strict_answer 가 True일 경우, LLM 답변 생성 없이 사전 정의된 답변을 제공하기 때문에 Hallucination 없는 답변을 신속하게 제공할 수 있습니다.
> 사전 정의된 답변 내용은 다음과 같습니다.
> `Unfortunately, we couldn't find any documents related to your query. Please try rephrasing or asking a different question.`

    - `source_visible`: Domain 정보를 클라이언트에게 전달할지 여부. 기본값은 `True`
    - `context_visible`: 답변에 활용된 Context를 클라이언트에게 전달할지 여부. 기본값은 `False`
    - `search_size`: Vector DB에서 검색된 문서 중 몇 건의 문서를 반환할지 설정. 기본값은 `10`
    - `context_size`: 검색된 문서 몇 개를 LLM으로 전달할지 설정. 기본값은 `3`
    - `min_similarity`: 검색된 문서의 최소 유사도 점수. 기본값은 `0.0`
    - `retrieval_type`: 검색 방식 (lexical, semantic, hybrid). 기본값은 `semantic`

- `demo`: RHEL AI Demo 관련 설정
    - `chat_playground`: Chat Playground 설정
        - `enabled`: 서버 기동 여부. 기본값은 `False`
        - `host`: 서버 IP 주소 또는 도메인명. 기본값은 `RHEL AI Pro Suite 실행 노드 IP`
        - `port`: 서버 포트. 기본값은 `8501`
    - `evaluation_dashboard`: Evaluation Dashboard 설정
        - `enabled`: 서버 기동 여부. 기본값은 `False`
        - `host`: 서버 IP 주소 또는 도메인명. 기본값은 `RHEL AI Pro Suite 실행 노드 IP`
        - `port`: 서버 포트. 기본값은 `8501`

> [!NOTE]
> Chat Playground와 Evaluation Dashboard의 port는 설정값이 우선합니다. 하지만 만약 port가 사용중일 경우,설정값부터 1씩 증가하며 사용가능한 port를 찾기 때문에 설정한 port 값과 실제 port값은 다를 수 있습니다. 실제 사용된 포트에 대한 정보는 생성시 출력된 log를 참조하십시오.

## 플러그인 설정

서버에서 사용하는 `Text Chunker`, `Embedding Model`, `Vector Store`, `LLM`, `Evaluator` 등 여러 모듈들은 프로젝트 내 플러그인 형태로 구현이 되어 있으며, 플러그인 최상위 디렉토리인 plugins 하위에 위치하고 있습니다. plugins 하위 1depth 디렉토리인 `chunker`, `embedding_model`, `evaluation`, `filter`, `llm`, `reranker`, `vector_store` 는 이름 변경이 불가하며, 해당 디렉토리 하위에 디렉토리 이름과 관련있는 플러그인들이 구현되어 있습니다. 해당 플러그인들을 서버에서 사용할 수 있도록 서버 실행시에 플러그인들을 로딩하고 있으며, 로딩시 필요한 정보들을 다음과 같이 설정하고 있습니다.

```yaml
plugins:
  chunker:
    recursive-character:
      chunk_size: 512
      chunk_overlap: 60
      tiktoken_encoding_name: cl100k_base

  filter:
    document:
      - path: context.metadata._source.metadata.status
        term: match #not_match, greater_than, less_than
        value:
          - trained

  embedding_model:
    huggingface-embedding:
      model_path: intfloat/multilingual-e5-large-instruct
      device: cpu

  vector_store:
    elasticsearch:
      hosts: [ "http://127.0.0.1:9200" ]
      index_name: ai_odyssey_demo_documents-000001
      embedding_model: huggingface-embedding

  reranker:
    bge-reranker:
      model_path: BAAI/bge-reranker-v2-m3

  # Example setup for LLM running on ollama
  llm:
    remote:
      host: http://127.0.0.1:8000
      model: merlinite-7b-lab-Q4_K_M.gguf

  evaluation:
    rhel-ai:
      llm: remote
      critic_llm:
        host: http://127.0.0.1:8000
        model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
      testset_index_name: ai_odyssey_demo_testset-000001
      evaluation_index_name: ai_odyssey_demo_evaluation-000001
      metrics:
        - faithfulness
        - answer_relevancy
        - context_precision
        - context_recall
        - context_entity_recall
        - noise_sensitivity_relevant
        - answer_similarity
```

- `chunker`: 텍스트 청킹(chunking) 플러그인 모음
    - `recursive-character`: 청크(chunk)가 충분히 작아질 때까지 주어진 문자 목록("\n\n", "\n", " ", "")의 순서대로 텍스트를 분할하는 chunker
        - `chunk_size`: 청크 size 정의
        - `chunk_overlap`: 청크 overlap 정의
        - `tiktoken_encoding_name`: 인코딩 모델 이름 정의

- `filter`: 검색 결과 필터링 플러그인 모음
    - `document`: 검색된 document를 필터링하는 플러그인
        - `path`: filter 대상 value를 가리키는 경로
        - `term`: filter 타입 (match, not_match, greater_than, less_than)
        - `value`: 사용자 정의 filter value

- `embedding_model`: 임베딩 모델 플러그인 모음
    - `huggingface-embedding`: `Huggingface`에 업로드되어 있는 Embedding Model을 이용하는 플러그인
        - `model_path`: Embedding Model 이름
        - `device`: 디바이스 설정 (cpu, cuda)
  
- `vector_store`: Vector DB 플러그인 모음
    - `elasticsearch`: Elasticsearch를 Vector DB로 이용하는 플러그인
        - `hosts`: Elasticsearch host들 정보
        - `index_name`: Elasticsearch에서 이용하는 document 저장 인덱스 이름
        - `embedding_model`: document 임베딩시에 사용할 모델 지정, 상기 `embedding_model` 플러그인에서 활용할 임베딩 모델 플러그인 지정

- `reranker`: 검색 결과 재정렬 플러그인 모음
    - `bge-reranker`: `BGE Reranker` 모델을 사용하여 document 순위를 조정하는 플러그인
        - `model_path`: document 순위 조정시에 사용할 모델 지정

- `llm`: LLM(Large Language Model) 관련 플러그인
    - `remote`: `llama.cpp`나 `vllm`으로 서빙되는 LLM과 통신하는 플러그인
        - `host`: 서버 접속 정보
        - `model`: llm 모델 이름

- `evaluation`: 검증 플러그인 모음
    - `rhel-ai`: RHEL-AI용 합성 데이터 생성 및 평가를 진행하는 플러그인
        - `llm`: 합성 데이터 생성 및 평가시 사용할 LLM 모델 지정. 상기 `llm` 플러그인에서 정의한 LLM 모델 플러그인 지정
        - `critic_llm`: 비평가 LLM 접속 정보
            - `host`: LLM 서버 접속 정보
            - `model`: 검증 대상 LLM 모델명
        - `testset_index_name`: Vector DB의 test set 저장 인덱스명
        - `evaluation_index_name`: Vector DB의 evaluation 결과 저장 인덱스명
        - `metrics`: 사용할 검증 지표(metric) 지정 (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `context_entity_recall`, `noise_sensitivity_relevant`, `answer_similarity`)

> [!NOTE]
> rhel-ai 플러그인은 [RAGAS v0.1.21](https://docs.ragas.io/en/v0.1.21/) 기반으로 구현되었습니다.
> 사용된 주요 지표들에 대한 설명은 [공식 문서](https://docs.ragas.io/en/v0.1.21/concepts/metrics/index.html)를 참조하십시오.
