# RHEL AI Pro Suite

## 소개

RedHat Enterprise Linux AI (RHEL AI)는 기업 환경에서 안정적이고 확장 가능한 AI 솔루션을 구축하기 위한 통합 플랫폼입니다. 이 플랫폼은 다양한 AI 워크로드를 지원하며, 기업은 이를 통해 데이터 분석, 머신 러닝, 딥 러닝, 컴퓨터 비전, 자연어 처리 등의 작업을 수행할 수 있습니다. RHEL AI는 안정적인 운영 환경을 제공하며, 기업은 이를 통해 AI 모델의 성능을 최적화하고 안정적인 서비스를 제공할 수 있습니다. 또한, RHEL AI는 오픈 소스 기술을 기반으로 하여 기업은 이를 통해 비용을 절감하고 혁신을 촉진할 수 있습니다.

하지만 RHEL AI에 기업 데이터를 적용하는 과정에는 여러 가지 어려움이 있습니다. 먼저 기업의 데이터를 RHEL AI가 학습할 수 있도록 yaml 형식으로 일일이 재구성해야 하는데, 이 과정에서 상당한 시간과 노력이 필요합니다. 또한 기업의 데이터는 매우 빠른 속도로 쌓이는 반면, LLM의 학습 속도가 이를 따라가지 못해 항상 최신 정보보다 뒤처지는 문제가 발생합니다. 게다가 현재 RHEL AI가 제공하는 검증 도구들은 일반적인 벤치마크 위주로 구성되어 있어서, LLM이 기업의 특수한 데이터를 얼마나 잘 학습했는지 정확하게 검증하기 어렵습니다.

RHEL AI를 더욱 쉽고 강력하게 활용할 수 있도록 RHEL AI Pro Suite 솔루션을 소개합니다.

## RHEL AI Pro Suite의 특장점

<img width="90%" src="docs/imgs/raps_key_features.png">

### 1. 학습 데이터 자동 생성
RHEL AI Pro Suite 솔루션은 RAG 기술을 기반으로 하며, 기업의 데이터를 Vector DB에 저장하면 자동으로 yaml 형식으로 변환해주는 기능을 제공합니다.
### 2. 지능적인 질의 처리
RHEL AI Pro Suite는 LLM이 이미 학습한 내용에 대해서는 context 없이 직접 질의하여 네트워크 비용을 줄이고 높은 성능을 제공합니다. 반면 LLM이 아직 학습하지 않은 내용에 대해서는 context를 포함하여 질의함으로써 정확도를 확보하고 최신 정보에 대응할 수 있습니다.
### 3. 기업 데이터에 특화된 검증
RHEL AI Pro Suite는 기업 데이터에 특화된 검증 도구를 제공하여, LLM이 기업 데이터를 얼마나 잘 학습했는지 정확하게 평가할 수 있습니다. 이러한 특징들을 통해 RHEL AI 기반의 RAG 솔루션으로서 차별화된 가치를 제공합니다.

## 시스템 요구 사항
- 32GB 이상의 메모리
- 500GB 이상의 스토리지
- Linux system (tested on Fedora)
- RHEL AI 1.1(InstructLab v0.18.4)
- Podman 최신 버전
- Elasticsearch 8.14 이상

## 설치
> [!NOTE] 
> 아래 설치 가이드는 CPU 기반의 장비에 RHEL AI, Elasticsearch, RHEL AI Pro Suite이 모두 설치되는 것을 가정합니다.

### 1. RHEL AI 1.1 설치
1. 만약 RHEL AI 1.1가 설치되어 있지 않다면, [RHEL AI 1.1 공식 설치 가이드](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.1/html-single/installing/index)를 참조하여 설치하십시오.

2. 설치 후 InsturctLab을 CPU 환경으로 초기화합니다.
    ```bash
    $ ilab config init
    Welcome to InstructLab CLI. This guide will help you to setup your environment.
    ...
    Please choose a train profile to use:
    [0] No profile (CPU-only)
    [1] A100_H100_x2.yaml
    [2] A100_H100_x4.yaml
    [3] A100_H100_x8.yaml
    [4] L40_x4.yaml
    [5] L40_x8.yaml
    [6] L4_x8.yaml
    Enter the number of your choice [hit enter for the default CPU-only profile] [0]:
    You selected: No profile (CPU-only)
    Initialization completed successfully, you're ready to start using `ilab`. Enjoy!
    ```

3. 본 예제에서는 `merlinite-7b-lab-Q4_K_M.gguf` 모델을 사용하기 때문에, 해당 모델의 리포지토리, 모델 및 Hugging Face 토큰을 지정합니다.
Hugging Face 토큰에 대한 자세한 내용은 [여기](https://huggingface.co/docs/hub/en/security-tokens)에서 확인할 수 있습니다.

    ```bash
    $ HF_TOKEN=<YOUR HUGGINGFACE TOKEN GOES HERE> ilab model download --repository -- instructlab/merlinite-7b-lab-GGUF --filename merlinite-7b-lab-Q4_K_M.gguf
    ```
    다운로드된 모델들은 아래와 같이 확인 할 수 있습니다.
    ```
    $ ilab model list
    +------------------------------+---------------------+--------+
    | Model Name                   | Last Modified       | Size   |
    +------------------------------+---------------------+--------+
    | merlinite-7b-lab-Q4_K_M.gguf | 2024-08-01 15:05:48 | 4.1 GB |
    +------------------------------+---------------------+--------+
    ```

4. 다음 커맨드를 통해 config.yaml 편집 모드로 진입합니다.
    ```bash
    $ ilab config edit
    ```
    아래 항목들을 merlinite-7b-lab-Q4_K_M.gguf 모델로 지정합니다.
    ```
    chat:
        model: ~/.cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf
    generate:
        model: ~/.cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf
        teacher:
            model_path: ~/.cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf
    serve:
        model_path: ~/.cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf

    ```

> [!NOTE]
> 해당 장비 사양에 맞게 generate.num_cpus 값(default: 10)을 수정하면 합성 데이터 생성시 더 좋은 성능을 기대할 수 있습니다.

### 2. Vector Database 설치
> [!IMPORTANT]
> RHEL AI Pro Suite는 현재 Elasticsearch만을 Vector DB로 지원합니다.

> [!NOTE]
> 아래 설치 가이드는 Podman용 Elasticsearch 8.15.4 버전을 기준으로 작성되었습니다. 보다 다양한 설치 방법은 [공식 가이드 문서](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html)를 참고하세요.

1. 만약 podman이 설치되어 있어 있지 않다면 container-tools meta-package를 설치합니다.
    ```bash
    $ sudo dnf install container-tools
    ```

2. Elasticsearch 이미지 다운로드 및 실행
    ```bash
    $ sudo podman run -d --name elasticsearch --memory 2048m \
        -p 9200:9200 -p 9300:9300 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        docker.elastic.co/elasticsearch/elasticsearch:8.15.4
    ```
> [!CAUTION]
> Elasticsearch를 상업용으로 활용하기 위해서는 라이선스 구매가 필요할 수 있습니다. 자세한 내용은 [공식 가이드 문서](https://www.elastic.co/subscriptions)를 참고하세요.

> [!CAUTION]
> 사용자 보안이 필요한 서비스는 xpack.security 설정이 필요합니다. 자세한 내용은 [공식 가이드 문서](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup-xpack.html)를 참고하세요.

3. 다음 명령을 통해 Elasticsearch 구동 여부를 확인할 수 있습니다.
    ```
    $ curl -XGET http://localhost:9200
    {
    "name" : "62a77893ad83",
    "cluster_name" : "docker-cluster",
    "cluster_uuid" : "sp1FzENTTV-0hgmKbUhBMQ",
    "version" : {
        "number" : "8.15.4",
        "build_flavor" : "default",
        "build_type" : "docker",
        "build_hash" : "4ec7e3608de63c104724277ebfa8dc7b84685f48",
        "build_date" : "2024-11-07T09:35:45.535387784Z",
        "build_snapshot" : false,
        "lucene_version" : "9.11.1",
        "minimum_wire_compatibility_version" : "7.17.0",
        "minimum_index_compatibility_version" : "7.0.0"
    },
    "tagline" : "You Know, for Search"
    }
    ```

### 3. RHEL AI Pro Suite 설치
1. Fedora Linux에 설치하는 경우 다음 명령을 실행하여 C++, Python 3.10 또는 3.11 및 기타 필요한 도구를 설치하십시오.
    ```bash
    $ sudo dnf install gcc gcc-c++ make git python3.11 python3.11-devel
    ```

2. RHEL AI Pro Suite repository clone
    ```bash
    $ cd ~
    $ git clone https://github.com/s-core/rhel-ai-pro-suite.git
    $ cd rhel-ai-pro-suite
    ```
    
3. 가상환경 구축 및 필요 모듈 설치
> [!CAUTION]
> RHEL AI Pro Suite의 python 가상환경과 RHEL AI 1.1의 ilab python 가상환경을 분리하십시오. 동일 환경으로 module들을 설치할 경우 충돌이 발생합니다.

    ```bash
    $ python3.11 -m venv --upgrade-deps raps
    $ source raps/bin/activate
    (raps) $ python -m pip install -r ./requirements.txt
    ```
> [!NOTE]
> The following steps in this document use Python venv for virtual environments. However, if you use another tool such as pyenv or Conda Miniforge for managing Python environments on your machine continue to use that tool instead. Otherwise, you may have issues with packages that are installed but not found in your virtual environment.

4. config/configuration_example.yml를 복사하여 config/configuration.yml 생성
    ```bash
    (raps) $ cp config/configuration_example.yml config/configuration.yml
    ```
5. (Optional) 만약 동일 장비가 아닌 별도 환경에서 RHEL AI와 Elasticsearch가 구동 중이라면, 해당 환경에 맞게 config/configuration.yml를 수정하십시오.

## Tutorial
> [!NOTE]
> RHEL AI 1.1과 RHEL AI Pro Suite는 병렬로 구동됩니다. 따라서 RHEL AI 1.1 조작용 터미널과 RHEL AI Pro Suite 조작용 터미널을 각각 활성화하시면 더 편리하게 시스템을 조작할 수 있습니다.

### 1. (데이터 삽입 전) RAG와 LLM의 답변 비교
1. 설치를 완료하셨다면 다음 명령을 통해 LLM Server를 구동합니다.
    ```bash
    $ ilab model serve
    ...
    INFO 2024-11-26 15:54:58,978 instructlab.model.backends.llama_cpp:194: After application startup complete see http://0.0.0.0:8000/docs for API.
    ```

2. (다른 터미널에서) RHEL AI Pro Suite의 RAG 서버를 구동합니다.
    ```bash
    $ cd ~/rhel-ai-pro-suite
    $ source raps/bin/activate
    (raps) $ python main.py
    ...
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
    ```
> [!NOTE]
> 최초 구동시, embedding model과 re-ranker model을 다운로드가 발생하여 수십분의 로딩시간이 발생할 수 있습니다.

3. 브라우저에서 http://localhost:8888/demo 에 접속 후 chat_playground를 선택합니다.

> [!NOTE]
> ~/rhel-ai-pro-suite/configs/configuration.yaml에 demo.chat_playground 설정이 변경되지 않았다면, http://localhost:8503 으로 바로 접속 가능합니다.

4. chat_playground에서 다음 질문을 입력하고 결과를 비교해봅니다.
    ```
    Tell me about K-pop artist Zerobaseone
    ```
    Zerobaseone은 유명한 K-POP 아티스트이지만 RAG는 대답을 못하고, LLM은 Hallucination을 발생시킬 것입니다.
    <img width="100%" src="docs/imgs/chat_without_data.png">

> [!NOTE]
> 대답 못하는 결과만 보고 RAG의 성능에 대해 의심할 것입니다. 하지만 RAG는 지극히 정상적으로 동작하고 있습니다. 이는 RAG가 지식 소스로 등록된 문서에서 관련 정보를 찾지 못했기 때문입니다. 이후 단계에서 지식 소스를 등록하고 다시 질문해보면 RAG가 정확한 답변을 제공할 것입니다.

### 2. (데이터 삽입 후) RAG와 LLM 간 답변 비교
1. (필요시 터미널을 더 띄워) 다음 명령을 통해 Data를 삽입해봅시다. 
    ```bash
    $ cd ~/rhel-ai-pro-suite
    $ curl -X 'POST' \
        'http://localhost:8888/v1/documents/files' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'file=@demo/sample_data/zb1.md' \
        -F 'domain=k-pop'
    
    # Result
    {"messsage":"success","index_name":"ai_odyssey_demo_documents-000001"}
    ```

2. 데이터가 잘 삽입되었는지는 Vector DB에서 확인 가능합니다.
    ```bash
    $ curl -XGET 'http://localhost:9200/ai_odyssey_demo_documents-000001/_search?size=0&pretty'
    
    # Result
    {
        ...
        "hits" : {
            ...
            "value" : 5,
            ...
        }
    }
    ```
    문서가 5개의 chunk로 나눠져서 잘 들어간 것을 확인할 수 있습니다.

3. chat_playground에서 이전과 동일한 질문을 입력하고 결과를 비교해봅니다.
    ```
    Tell me about K-pop artist Zerobaseone
    ```
    RAG는 Vector DB에 삽입된 지식 소스 기반으로 답변을 잘 생성하며, LLM은 여전히 Hallucination이 발생하는 것을 확인 할 수 있습니다. 

    <img width="100%" src="docs/imgs/chat_with_contexts_off.png">

    답변 아래 Context 버튼을 눌러 확인해보면 답변에 기반이 되는 지식을 확인 할 수 있습니다.

    <img width="100%" src="docs/imgs/chat_with_contexts_on.png">

> [!NOTE]
> RAG의 가치는 데이터만 있으면 즉각적으로 보다 정확한 답변을 생성할 수 있다는데 있습니다. 하지만 프롬프트 상에 많은 데이터가 전달되어 네트워크 비용과 LLM 연산 증가를 발생시키는 단점도 존재합니다. LLM은 이러한 질문에 대답하기 위해서는 학습이 필요합니다.

### 3. LLM 학습용 데이터 만들기
#### RHEL AI Pro Suite로 qna.yaml 생성하기 
RHEL AI 학습을 위한 taxonomy tree를 만드는 것은 원래 많은 시간과 노력이 필요한 작업입니다. 하지만 RHEL AI Pro Suite를 사용하면 이 과정을 크게 단순화할 수 있습니다.
이 도구는 Vector DB의 데이터를 활용해서 taxonomy에 필요한 qna.yaml 파일을 자동으로 생성합니다. 사용자는 두 가지를 먼저 결정해야 됩니다.

* domain - 어떤 영역의 데이터를 생성할지
* testset_size - 몇 개의 질문/답변 쌍을 만들지

그리고 나서 RHEL AI의 texonomy 요건을 만족할 조건들을 추가 입력해야 합니다.

* qna_yaml - qna.yaml에 지켜야하는 요건에 대한 내용입니다.
    * version - "3"으로 고정되어 있습니다.
    * created_by - 작성자 정보입니다.
    * repo, commit, patterns - 데이터가 올라가 있는 repository, commit, 파일명 정보입니다.

> [!IMPORTANT]
> RHEL AI 1.1용 학습 데이터는 Git 저장소에 호스팅되어야 합니다. [공식 문서](https://docs.redhat.com/ko/documentation/red_hat_enterprise_linux_ai/1.1/html/creating_a_custom_llm_using_rhel_ai/customize_taxonomy_tree#adding_knowledge)를 참조하세요.

이렇게 API를 호출하면 ilab에서 자동으로 합성 데이터가 포함된 qna.yaml 파일을 생성합니다. 사용자는 복잡한 제약조건을 고려해서 직접 파일을 만들 필요 없이, 생성된 파일을 taxonomy의 적절한 위치에 배치하기만 하면 됩니다.

```bash
$ mkdir -p ~/.local/share/instructlab/taxonomy/knowledge/arts/music/k-pop
$ curl -X POST http://localhost:8888/v1/qna/generate -H "Content-Type: application/json" -d '{
    "target_model": "", 
    "testset_size": 3,
    "domain": "k-pop",
    "document_index_name": "ai_odyssey_demo_documents-000001",
    "testset_index_name": "ai_odyssey_demo_testset-000001",
    "qna_yaml": {
        "version": 3,
        "created_by": "test_author",
        "repo": "https://github.com/s-core/rhel-ai-pro-suite",
        "commit": "5f7158a5ce83c4ff493bfe341fe31ecad64ff697",
        "patterns": ["zb1.md"]
    }
}' > ~/.local/share/instructlab/taxonomy/knowledge/arts/music/k-pop/qna.yaml

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   411    0     0    0   411      0      0 --:--:--  1:50:30 --:--:--     0
```


> [!IMPORTANT]
> 현재 Taxonomy에서 Skill 영역은 지원하지 않으며, 모든 데이터는 Knowledge 영역에 위치해야 합니다.
> Vector DB 안에 적어도 5개 이상의 chunk가 존재해야 하며, testset_size 값은 3 이상이어야 합니다.

> [!WARNING]
> 해당 과정은 28코어 CPU 장비 기준으로 2시간 정도 소요됩니다.

> [!NOTE]
> RHEL AI의 Taxonomy에 대해 보다 자세한 내용은 [공식 문서](https://docs.redhat.com/ko/documentation/red_hat_enterprise_linux_ai/1.1/html/creating_a_custom_llm_using_rhel_ai/customize_taxonomy_tree#customize_taxonomy_tree)를 참조하세요.

#### InsturctLab으로 taxonomy 검증하기
이제 생성된 yaml을 다음 명령을 통해 검증할 수 있습니다.
```bash
$ ilab taxonomy diff
knowledge/arts/music/k-pop/qna.yaml
Taxonomy in /home/rocky/.local/share/instructlab/taxonomy is valid :)
```
> [!NOTE]
> 양자화 모델를 활용하여 qna.yaml을 생성할 경우, 수정사항이 발생할 수 있습니다. 이 경우, 가이드에 따라 qna.yaml을 수정하고 다시 검증해야 합니다.

#### InsturctLab으로 합성 데이터 생성하기
RHEL AI는 qna.yaml을 기반으로 LLM이 학습할 수 있는 합성 데이터를 생성해냅니다.
이를 통해 상대적으로 적은 수의 qna.yaml 만으로 LLM 학습 품질을 높이는 것이 가능합니다.
다음은 CPU 환경을 고려하여 teacher 모델없이 합성 데이터를 적게 만들어 빠르게 합성 데이터를 생성하는 명령입니다.

```bash
$ ilab data generate --pipeline simple --sdg-scale-factor 5 --enable-serving-output 
```
> [!NOTE]
> 좋은 Teacher 모델을 사용하고, 합성 데이터를 많이 생성할수록 더 나은 LLM 학습 품질을 기대할 수 있습니다. 보다 자세한 내용은 [공식 문서](https://docs.redhat.com/ko/documentation/red_hat_enterprise_linux_ai/1.1/html/creating_a_custom_llm_using_rhel_ai/generate_sdg)를 참조하세요.

> [!WARNING]
> 해당 과정은 28코어 CPU 장비 기준으로 8시간 이상 소요됩니다.

### 4. InstuctLab으로 LLM 학습시키기
InstuctLab에서는 생성된 합성 데이터를 활용하여 LLM의 학습을 지원합니다.
다음은 CPU 환경을 고려하여 최소한으로 학습을 수행하는 명령입니다.
```bash
$ ilab model train --legacy --iters 1 --num-epochs 1 --enable-serving-output
```
> [!WARNING]
> 해당 과정은 28코어 CPU 장비 기준으로 48 시간 이상 소요됩니다.

학습 이후 다음 명령을 통해 새로 생성된 모델에 대해 테스트를 진행할 수 있습니다.
```bash
$ ilab model test
```
### 5. 데이터 학습 여부 체크하기
#### LLM에 학습 검증하기
이제 사람이 나서야 될 차례입니다. 다음과 같은 방법을 통해 학습된 LLM이 우리가 원하는 대로 학습되었는지 확인할 수 있습니다.

1. cli chat 환경에서 zerobaseone에 대해 물어보기
```bash
$ ilab model chat --model ~/.local/share/instuctlab/checkpoints/ggml-model-f16.gguf
```
2. chat_playground에서 zerobaseone에 대해 물어보기
```bash
$ ilab model serve --model-path ~/.local/share/instuctlab/checkpoints/ggml-model-f16.gguf
```

#### InstuctLab에서 사용 모델 변경하기
학습이 되었는지 직접 확인하셨다면, 이제 새로운 모델을 선보일 차례입니다.
다음 절차를 수행하셔서 새로운 모델을 서비스 하실 수 있습니다.

1. 학습된 LLM으로 모델 전환
```bash
$ ilab model convert
```

2. 학습된 LLM 서빙
```bash
$ ilab model serve --model-path instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf
```

#### RHEL AI Pro Suite를 통해 필요없는 context 필터링하기
학습된 LLM이 적용되었지만 여전히 RAG는 질의시마다 수많은 context를 전송하고 있습니다.
네트워크 비용이나 LLM의 최대 처리 token 수를 생각하면 참으로 유감스러운 일입니다.
다음 API를 호출하여 Vector DB의 해당 domain 지식들을 필터링합시다.

```bash
curl -X PATCH http://localhost:8888/v1/documents \
    -H "Content-Type: application/json" -d '{
        "filter": {
            "domain": "k-pop"
        },
        "update": {
            "status": "trained"
        }
    }'
```

이제 RAG는 질의에 대해 Vector DB에서 검색을 하더라도, 이미 학습한 내용에 대해서는 LLM에 전달하지 않습니다. Context 없이 답변을 잘하는지 여부는 chat_playground에서 손쉽게 확인 가능합니다.

### 6. 데이터와 모델 검증하기


```bash
curl -X 'POST' 'http://localchost:8888/v1/qna/evaluate' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
    "k": 1,
    "domain": "k-pop"
}
```

API 호출 후에는 evaluation_dashboard를 실행하여 각종 메트릭을 확인 가능합니다.

<img width="100%" src="docs/imgs/evaluation.png">


## API Endpoints
API에 대한 내용은 [여기](docs/rest_api.md)를 참조하십시오.

## License
RHEL AI Pro Suite is distributed under AGPL-3.0.
