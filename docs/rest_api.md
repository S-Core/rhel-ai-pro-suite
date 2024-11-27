# RHEL AI Pro Suite APIs

## POST `/v1/documents/texts`
전달되는 document(text)를 VectorStore에 색인하는 API

### Request
#### Body
- `content`: (Required) 청크 및 임베딩 변환이 될 원본 텍스트
- `metadata`: (Required) document 메타 데이터
  - `source`: (Optional) 텍스트 출처
  - `name`: (Optional) document 이름
  - `domain`: (Required) taxonomy 분야
  - `author`: (Optional) 작성자
- `index_name`: (Optional) 색인 대상 인덱스 이름. Default: `설정 파일내 vector_store의 index_name`

#### Example
```bash
curl -X POST http://localhost:8888/v1/documents/texts -H "Content-Type: application/json" -d "{
    \"content\": \"Zerobaseone (RR: Jerobeiseuwon; stylized in all caps; abbreviated as ZB1) is a South Korean boy band formed through Mnet's reality ...\",
    \"metadata\": {
        \"source\": \"https://test-example.com/10\",
        \"name\": \"k-pop singer\",
        \"domain\": \"k-pop\",
        \"author\" : \"Michael\"
    },
    \"index_name\": \"ai_odyssey_demo_documents-000001\"
}"
```

### Response
#### Success (200)
```json
{
    "message": "success",
    "index_name": "ai_odyssey_demo_documents-000001"
}
```

#### Failure (500)
```json
{
    "TID": "8ca96e7d-9d05-477c-81bb-9aae2d91a1d2",
    "error code": 11999,
    "message": "Chunker Fail",
    "detail": "chunk exception"
}

```

## POST `/v1/documents/files`
전달되는 document(file)를 VectorStore에 색인하는 API

### Request
#### Parameter
- `index_name`: (Optional) 색인 대상 인덱스 이름. Default: `설정 파일내 vector_store의 index_name`

#### Body
- `file`: (Required) text 형태의 파일
- `source`: (Optional) 텍스트 출처
- `name`: (Optional) document 이름
- `domain`: (Required) taxonomy 분야
- `author`: (Optional) 작성자

#### Example
```bash
curl -X POST http://localhost:8888/v1/documents/files?index_name=ai_odyssey_demo_documents-000001 \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/your/file.txt" \
    -F "source=https://test-example.com/10" \
    -F "name=k-pop singer" \
    -F "domain=k-pop" \
    -F "author=Michael"
```

### Response

#### Success (200)
```json
{
    "message": "success",
    "index_name": "ai_odyssey_demo_documents-000001"
}
```

#### Failure (500)
```json
{
    "TID": "8ca96e7d-9d05-477c-81bb-9aae2d91a1d2",
    "error code": 11999,
    "message": "Chunker Fail",
    "detail": "chunk exception"
}
```

## POST `/v1/qna/generate`
QnA 테스트 셋 생성 및 qna.yaml 파일 내용을 전달하는 API

### Request
#### Body
- `target_model`: (Optional) QnA 테스트 셋을 생성할 LLM. Default: `설정 파일내 evaluation의 model`
- `testset_size`: (Optional) 조회된 Content 1개당 생성할 QnA 테스트 셋 개수. Default: `3`
- `domain`: (Required) taxonomy 분야
- `document_index_name`: (Optional) 컨텐츠 조회 대상 인덱스 이름. Default: `설정 파일내 vector_store의 index_name`
- `testset_index_name`: (Optional) QA 테스트 셋 색인 대상 인덱스 이름. Default: `설정 파일내 evaluation의 testset_index_name`
- `qna_yaml`: (Required) QnA yaml 파일 설정
  - `version`: (Optional) 버전. Default: `3`
  - `created_by`: (Required) Git 사용자 이름
  - `document_outline`: (Optional) 문서의 개요. Default: `LLM을 통해 컨텐츠를 요약`
  - `repo`: (Required) 마크다운 파일을 보관하는 repository의 URL
  - `commit`: (Required) 마크다운 파일과 함께 리포지토리에 있는 커밋의 SHA
  - `patterns`: (Required) 리포지토리에 있는 마크다운 파일을 지정

#### Example
```bash
curl -X POST http://localhost:8888/v1/qna/generate -H "Content-Type: application/json" -d "{
    \"target_model\": \"/home/[user-id]/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf\",
    \"testset_size\": 3,
    \"domain\": \"k-pop\",
    \"document_index_name\": \"ai_odyssey_demo_documents-000001\",
    \"testset_index_name\": \"ai_odyssey_demo_testset-000001\",
    \"qna_yaml\": {
        \"version\": 3,
        \"created_by\": \"test_author\",
        \"repo\": \"https://github.com/juliadenham/Summit_knowledge\",
        \"commit\": \"5f7158a5ce83c4ff493bfe341fe31ecad64ff697\",
        \"patterns\": [\"chickadee.md\"]
    }
}"
```

### Response:
#### Success (200)
```yaml
version: 3
created_by: test_author
domain: k-pop
seed_examples:
- context: 'Zerobaseone (RR: Jerobeiseuwon; stylized in all caps; abbreviated as ZB1) is a South Korean
    boy band formed through Mnet''s reality competition program Boys Planet and managed by WakeOne. The
    group consists of nine members: Kim Ji-woong, Zhang Hao, Sung Han-bin, Seok Matthew, Kim Tae-rae,
    Ricky, Kim Gyu-vin, Park Gun-wook, and Han Yu-jin. ...'
  questions_and_answers:
  - question: ' "Which South Korean boy band was formed through Mnet''s reality competition program Boys
      Planet and achieved commercial success with their debut EP?'
    answer: Zerobaseone
  - question: ' "Which South Korean boy band, formed through Mnet''s Boys Planet competition and managed
      by WakeOne, debuted in 2023 with the EP ''Youth in the Shade'' and achieved commercial success?'
    answer: Zerobaseone
  - question: ' Who are the nine members of Zerobaseone, a South Korean boy band formed through Mnet''s
      Boys Planet competition?'
    answer: 'The nine members of Zerobaseone are: Kim Ji-woong, Zhang Hao, Sung Han-bin, Seok Matthew,
      Kim Tae-rae, Ricky, Kim Gyu-vin, Park Gun-wook, and Han Yu-jin.'
- context: ...
document_outline: ' South Korean boy band Zerobaseone (ZB1), formed through Mnet''s reality competition
  Boys Planet, debuted with EP "Youth in the Shade" in July 2023. The group of nine members, including
  Kim Ji-woong and previous contestants on other shows, sold over two million units, topped South Korean
  charts, and earned global recognition with Billboard Global 200 hit "In Bloom."'
document:
  repo: https://github.com/juliadenham/Summit_knowledge
  commit: 5f7158a5ce83c4ff493bfe341fe31ecad64ff697
  patterns:
  - chickadee.md
```

#### Failure (500)
```json
{
    "TID": "8ca96e7d-9d05-477c-81bb-9aae2d91a1d2",
    "error code": 40005,
    "message": "Testset Result Generation Fail!",
    "detail": "testset result exception"
}
```

## PATCH `/v1/documents`
Document 메타데이터 업데이트

### Request
#### Parameter
- `index_name`: (Optional) 색인 대상 인덱스 이름. Default: `설정 파일내 vector_store의 index_name`

#### Body
- `filter`:
  - `domain`: (Required) document 유형
- `update`:
  - `status`: (Required) document 상태값. 현재는 "trained"만 지원

#### Example
```bash
curl -X PATCH http://localhost:8888/v1/documents -H "Content-Type: application/json" -d "{
    \"filter\": {
        \"domain\": \"k-pop\"
    },
    \"update\": {
        \"status\": \"trained\"
    }
}"
```

### Response

#### Success (200)
```json
{
    "messsage": "success",
    "updated_count": 5
}
```

#### Failure (500)
```json
{
  "TID": "6b36673d-ff2e-493c-8e00-ebed52dc5721",
  "error code": 20999,
  "message": "VectorStore Process Fail",
  "detail": "NotFoundError(404, 'index_not_found_exception', 'no such index [aaaa]', aaaa, index_or_alias)"
}
```

## POST `/v1/qna/evaluate`
QnA 테스트 셋 평가 및 평가 결과를 VectorStore에 색인하는 API

### Request
#### Body
- `target_model`: (Optional) 답변(answer)을 생성할 LLM. Default: `설정 파일내 evaluation의 model`
- `k`: (Optional) RAG 파이프라인의 context 사이즈. Default: `3`
- `domain`: (Required) taxonomy 분야
- `testset_index_name`: (Optional) QnA 테스트 셋 조회 대상 인덱스 이름. Default: `설정 파일내 evaluation의 testset_index_name`
- `evaluation_index_name`: (Optional) 생성된 평가 결과 색인 대상 인덱스 이름. Default: `설정 파일내 evaluation의 evaluation_index_name`

#### Example
```bash
curl -X POST http://localhost:8888/v1/qna/evaluate -H "Content-Type: application/json" -d "{
    \"target_model\": \"/home/[user-id]/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf\",
    \"k\": 1,
    \"domain\": \"k-pop\",
    \"testset_index_name\": \"ai_odyssey_demo_testset-000001\",
    \"evaluation_index_name\": \"ai_odyssey_demo_evaluation-000001\"
}"
```

### Response
#### Success (200)
```json
{
    "status": "ok",
    "message": {
        "data": [
            {
                "question": " Which South Korean boy band was formed through Mnet's reality competition program Boys Planet and debuted with commercial success in 2023?",
                "ground_truth": "Zerobaseone",
                "answer": " Zerobaseone is a South Korean boy band formed through Mnet's reality competition program Boys Planet and debuted with commercial success in 2023, selling over two million units and peaking number one on South Korean Circle Album Chart. The group consists of nine members: Kim Ji-woong, Zhang Hao, Sung Han-bin, Seok Matthew, Kim Tae-rae, Ricky, Kim Gyu-vin, Park Gun-wook, and Han Yu-jin.",
                "contexts": [
                    "Zerobaseone (RR: Jerobeiseuwon; stylized in all caps; abbreviated as ZB1) is a South Korean boy band formed through Mnet's reality competition program Boys Planet and managed by WakeOne. The group consists of nine members: Kim Ji-woong, Zhang Hao, Sung Han-bin, Seok Matthew, Kim Tae-rae, Ricky, Kim Gyu-vin, Park Gun-wook, and Han Yu-jin. ..."
                ],
                "metadata": {
                    "source": "https://test-example.com/10",
                    "name": "k-pop singer",
                    "domain": "k-pop",
                    "author": "Michael",
                    "referenceContexts": [
                        "Zerobaseone (RR: Jerobeiseuwon; stylized in all caps; abbreviated as ZB1) is a South Korean boy band formed through Mnet's reality competition program Boys Planet and managed by WakeOne. The group consists of nine members: Kim Ji-woong, Zhang Hao, Sung Han-bin, Seok Matthew, Kim Tae-rae, Ricky, Kim Gyu-vin, Park Gun-wook, and Han Yu-jin. ..."
                    ],
                    "generatorLLM": "/home/[user-id]/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    "criticLLM": "/home/[user-id]/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    "timestamp": 1732264019.01063
                },
                "scores": {
                    "faithfulness": 0.9163240208552347,
                    "answer_relevancy": 0.8363240208552298,
                    "context_precision": -1.0,
                    "context_recall": 0.7393248506554236,
                    "context_entity_recall": 0.9999999900000002,
                    "noise_sensitivity_relevant": 0.0,
                    "answer_similarity": 0.8493240208552298
                }
            },
            ...
        ],
        "target_model": "/home/[user-id]/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "critic_model": "/home/[user-id]/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    }
}
```

#### Failure (500)
```json
{
    "TID":"175ef959-d97f-4641-a4b0-5b78fb1c021b",
    "error code":41999,
    "message":"Evaluation Fail!",
    "detail":"evaluation exception"}
```