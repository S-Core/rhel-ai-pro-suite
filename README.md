# RHEL AI Pro Suite

English | [한국어](./i18n/README-KR.md)

## Introduction

RedHat Enterprise Linux AI (RHEL AI) is a unified platform for building reliable and scalable AI solutions in enterprise environments. It supports a wide range of AI workloads, including data analytics, machine learning, deep learning, computer vision, and natural language processing, and provides a stable operating environment to optimize the performance of AI models and deliver high-quality services. It is also based on open source technologies, which makes it cost-effective and accelerates innovation.

However, there are a few challenges to actually using RHEL AI in the enterprise. For starters, converting an organization's data into a yaml format that RHEL AI can learn from is time-consuming and resource-intensive, and while an organization's data accumulates rapidly in real time, LLM's learning speed can't keep up, leaving a gap in the latest information. Additionally, the current validation tools provided by RHEL AI are organized around generic benchmark tests, making it difficult to assess how accurately LLM has learned your organization's unique data.

This demo will showcase the new RHEL AI Pro Suite, which overcomes these limitations and helps you get more out of RHEL AI.


## Features of RHEL AI Pro Suite

<img width="90%" src="docs/imgs/raps_key_features.png">

### 1. Automatically generate training data
The RHEL AI Pro Suite solution is based on RAG technology and provides the ability to automatically convert corporate data into yaml format when stored in Vector DB.
### 2. Intelligent query processing
RHEL AI Pro Suite directly queries LLM without context about what it has already learned, reducing network costs and providing high performance. On the other hand, querying LLM with context about what it hasn't yet learned ensures accuracy and allows it to respond to the latest information.
### 3. Validation specialized for enterprise data
RHEL AI Pro Suite provides enterprise data-specific validation tools so you can accurately assess how well LLM has learned your enterprise data. These features provide differentiated value as a RAG solution based on RHEL AI.

## System Requirements
- 32GB or more of memory
- 500GB or more of storage
- Linux system (tested on Fedora)
- InstructLab v0.21.0(RHEL AI 1.3)
- Podman latest version
- Elasticsearch 8.14 or later

## Installation
> [!NOTE] 
> The installation guide below assumes that RHEL AI, Elasticsearch, and RHEL AI Pro Suite are all installed on a single CPU-based machine.

### 1. InstructLab Install
1. If you don't have InstructLab installed, see the [InstructLab installation guide](https://github.com/instructlab/instructlab?tab=readme-ov-file#-getting-started) or the [RHEL AI installation guide](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.2/html-single/installing/index) to install it.

2. After installation, initialize InsturctLab to the CPU environment.
    ```bash
    (venv) $ ilab config init
    ```
    Below is an example of the log output after the initialization command.
    ```
    ----------------------------------------------------
            Welcome to the InstructLab CLI
    This guide will help you to setup your environment
    ----------------------------------------------------

    Please provide the following values to initiate the environment [press Enter for defaults]:
    Path to taxonomy repo [/home/user/.local/share/instructlab/taxonomy]:
    Path to your model [/home/user/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf]:

    Generating config file and profiles:
        /home/user/.config/instructlab/config.yaml
        /home/user/.local/share/instructlab/internal/system_profiles

    We have detected the INTEL CPU profile as an exact match for your system.

    --------------------------------------------
        Initialization completed successfully!
      You're ready to start using `ilab`. Enjoy!
    --------------------------------------------
    ```

3. Since this example uses the models `granite-7b-lab-Q4_K_M.gguf`, `mistral-7b-instruct-v0.2.Q4_K_M.gguf`, and `instructlab/granite-7b-lab`, you specify the repository, model, and Hugging Face token for those models.
You can learn more about Hugging Face tokens [here](https://huggingface.co/docs/hub/en/security-tokens).

    ```bash
    (venv) $ ilab model download
    (venv) $ HF_TOKEN=<YOUR HUGGINGFACE TOKEN GOES HERE> ilab model download --repository -- instructlab/granite-7b-lab
    ```
    The downloaded models can be found below.
    ```
    (venv) $ ilab model list
    +--------------------------------------+---------------------+---------+
    | Model Name                           | Last Modified       | Size    |
    +--------------------------------------+---------------------+---------+
    | granite-7b-lab-Q4_K_M.gguf           | 2024-12-02 21:09:02 | 3.8 GB  |
    | merlinite-7b-lab-Q4_K_M.gguf         | 2024-12-02 21:37:35 | 4.1 GB  |
    | instructlab/granite-7b-lab           | 2024-12-03 11:07:34 | 12.6 GB |
    | mistral-7b-instruct-v0.2.Q4_K_M.gguf | 2024-11-07 16:24:00 | 4.1 GB  |
    +--------------------------------------+---------------------+---------+
    ```

1. Enter config.yaml edit mode with the command
    ```bash
    (venv) $ ilab config edit
    ```
    In the configuration, make sure that the model is specified as below for each action.
    ```yaml
    chat:
        model: ~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf
    generate:
        model: ~/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
        teacher:
            model_path: ~/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
    serve:
        model_path: ~/.cache/instructlab/models/granite-7b-lab-Q4_K_M.gguf
    train:
        model_path: ~/.cache/instructlab/models/instructlab/granite-7b-lab
    ```

> [!NOTE]
> Modify the value of generate.num_cpus (Default: 10) to match your instrument specification and you can expect better performance when generating synthetic data.

### 2. Vector Database Install

> [!IMPORTANT]
> RHEL AI Pro Suite currently only supports Elasticsearch as a Vector DB.

> [!NOTE]
> The installation guide below is based on Elasticsearch for Podman version 8.15.4. For more installation instructions, please refer to the [official guide documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html).

1. If you don't have podman installed, install the container-tools meta-package.
    ```bash
    $ sudo dnf install container-tools
    ```

2. Download and run the Elasticsearch image
    ```bash
    $ podman pull docker.elastic.co/elasticsearch/elasticsearch:8.15.4
    $ podman run -d --name elasticsearch --memory 2048m -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.15.4
    ```

3. You can verify that Elasticsearch is running with the following command.

    ```bash
    $ curl -XGET http://localhost:9200
    ```

    If you get a response like the one below, Elasticsearch is running fine.

    ```json
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

> [!CAUTION]
> Commercial use of Elasticsearch may require the purchase of a license. For more information, see the [official guide documentation](https://www.elastic.co/subscriptions).

> [!CAUTION]
> Services that require user security require the xpack.security setting. For more information, see the [official guide documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/security-settings.html#general-security-settings).

### 3. RHEL AI Pro Suite Install

> [!CAUTION]
> Separate the python virtual environment of RHEL AI Pro Suite from the ilab python virtual environment of RHEL AI. Installing modules in the same environment will cause conflicts.

> [!NOTE]
> The following steps in this document use Python venv for virtual environments. However, if you use another tool such as pyenv or Conda Miniforge for managing Python environments on your machine continue to use that tool instead. Otherwise, you may have issues with packages that are installed but not found in your virtual environment.

1. If you are installing on Fedora Linux, run the following commands to install C++, Python 3.10 or 3.11, and any other necessary tools.

    ```bash
    sudo dnf install gcc gcc-c++ make git python3.11 python3.11-devel git-lfs yq
    ```

2. Clone the RHEL AI Pro Suite repository

    ```bash
    $ cd ~
    $ git clone https://github.com/s-core/rhel-ai-pro-suite.git
    $ cd rhel-ai-pro-suite
    ```
    
3. Create a virtual environment and install the required modules

    ```bash
    $ python3.11 -m venv --upgrade-deps raps
    $ source raps/bin/activate
    (raps) $ python -m pip install -r ./requirements.txt
    ```

4. Copy config/configuration_example.yml to create config/configuration.yml

    ```bash
    (raps) $ cp config/configuration_example.yml config/configuration.yml
    ```

5. (Optional) If you are running RHEL AI and Elasticsearch in separate environments (not on the same machine), modify config/configuration.yml to match your environment.

## Tutorial

> [!NOTE]
> RHEL AI and RHEL AI Pro Suite run in parallel, so this tutorial requires you to enable the RHEL AI operator terminal and the RHEL AI Pro Suite operator terminal to operate the system more conveniently.

### 1. Comparing answers from RAG and LLM (before inserting data)
1. Once you have completed the installation, run the following command to start LLM Server.

    ```bash
    (venv) $ ilab model serve
    ```

    If the following message is displayed, it is successfully started.

    ```
    ...
    INFO 2024-11-26 15:54:58,978 instructlab.model.backends.llama_cpp:194: After application startup complete see http://127.0.0.1:8000/docs for API.
    ```

2. Start the RAG server of RHEL AI Pro Suite (in another terminal).

    ```bash
    $ cd ~/rhel-ai-pro-suite
    $ source raps/bin/activate
    (raps) $ python main.py
    ```

    If the following message is displayed, it is successfully started.

    ```
    ...
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
    ```

3. In your browser, go to http://localhost:8888/demo and select chat_playground.

4. In chat_playground, type the following questions and compare the results.

    ```
    Tell me about K-pop artist Zerobaseone
    ```

    Zerobaseone is a famous K-POP artist, but RAG will not be able to answer, and LLM will cause Hallucination.

    <img width="100%" src="docs/imgs/chat_without_data.png">

> [!NOTE]
> If the demo.chat_playground setting in ~/rhel-ai-pro-suite/configs/configuration.yaml has not been changed, you can access it directly at http://localhost:8503

> [!NOTE]
> On the first run, the embedding model and re-ranker model will be downloaded, which can take tens of minutes of loading time.

> [!NOTE]
> You might be tempted to question RAG's performance based on the unanswered questions, but RAG is working perfectly fine. This is because RAG did not find the relevant information in the documentation that you have registered as a knowledge source. At a later stage, you can register the knowledge source and ask the question again, and RAG will provide the correct answer.

### 2. Comparing answers from RAG and LLM (after inserting data)
1. Let's insert Data with the following command (open more terminals if necessary) 

    ```bash
    $ cd ~/rhel-ai-pro-suite
    $ curl -X 'POST' \
        'http://localhost:8888/v1/documents/files' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'file=@demo/sample_data/zb1.md' \
        -F 'domain=k-pop'
    ```

    The result should look like this

    ```
    {"messsage":"success","index_name":"ai_odyssey_demo_documents-000001"}
    ```

2. You can check if the data was successfully inserted in Vector DB.

    ```bash
    $ curl -XGET 'http://localhost:9200/ai_odyssey_demo_documents-000001/_search?size=0&pretty'
    ```

    The result should look like this

    ```
    {
        ...
        "hits" : {
            ...
            "value" : 5,
            ...
        }
    }
    ```

    You can see that the document is split into 5 chunks and inserted successfully.

3. In chat_playground, enter the same question as before and compare the results.
    ```
    Tell me about K-pop artist Zerobaseone
    ```
    You can see that RAG does a good job of generating answers based on the knowledge sources inserted into the Vector DB, while LLM still experiences hallucinations.

    <img width="100%" src="docs/imgs/chat_with_contexts_off.png">

    Click the Context button below the answer to see the knowledge behind the answer.

    <img width="100%" src="docs/imgs/chat_with_contexts_on.png">

> [!NOTE]
> The value of RAGs is that they can generate more accurate answers on the fly, as long as data is available. However, the downside is that a lot of data is passed over the prompts, which increases network cost and LLM computation. LLM needs to be trained to answer these questions.

### 3. Creating data for LLM training
#### Generating qna.yaml with RHEL AI Pro Suite 
Creating a taxonomy tree for RHEL AI training is inherently a time-consuming and laborious task. However, the process can be greatly simplified with RHEL AI Pro Suite.
The tool leverages data from Vector DB to automatically generate the qna.yaml file needed for the taxonomy. The user needs to decide two things first.

* domain - Which areas of data to generate
* testset_size - How many question/answer pairs to create

Then you need to provide additional criteria to satisfy RHEL AI's taxonomy requirements.

* qna_yaml - About the requirements that must be followed in qna.yaml.
    * version - It is fixed at “3”.
    * created_by - Author information.
    * repo, commit, patterns - The repository, commit, and filename information where the data is located.

> [!IMPORTANT]
> Training data for InstructLab should be hosted in a Git repository, see the [official documentation](https://docs.redhat.com/ko/documentation/red_hat_enterprise_linux_ai/1.2/html/creating_a_custom_llm_using_rhel_ai/customize_taxonomy_tree#adding_knowledge).

When you call the API in this way, ilab will automatically generate a qna.yaml file containing the synthesized data. You don't need to create the file yourself, taking into account complex constraints, just place the generated file in the appropriate place in your taxonomy.

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
        "created_by": "wonseop",
        "repo": "https://github.com/s-core/rhel-ai-pro-suite",
        "commit": "f2975127aff4ce301c47d24a9a42e7865caa17b8",
        "patterns": ["demo/sample_data/zb1.md"]
    }
}' > ~/.local/share/instructlab/taxonomy/knowledge/arts/music/k-pop/qna.yaml
```

The result should look like this

```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 13633    0 13197    0   436      4      0 --:--:--  0:45:43 --:--:--  3
```

> [!IMPORTANT]
> Currently, Taxonomy does not support the Skill domain, and all data must be located in the Knowledge domain.
> At least 5 chunks must exist in the Vector DB, and the testset_size value must be at least 3.


> [!WARNING]
> The process takes about 1 hour on an Intel(R) Core(TM) i7-14700 CPU.

> [!NOTE]
> For more information about Taxonomy in RHEL AI, see the [official documentation](https://docs.redhat.com/ko/documentation/red_hat_enterprise_linux_ai/1.2/html/creating_a_custom_llm_using_rhel_ai/customize_taxonomy_tree#customize_taxonomy_tree).

#### Validating your taxonomy with InsturctLab
You can now validate the generated YAML with the following command

```bash
$ ilab taxonomy diff
knowledge/arts/music/k-pop/qna.yaml
Taxonomy in /home/rocky/.local/share/instructlab/taxonomy is valid :)
```

> [!IMPORTANT]
> When generating qna.yaml using the quantization model, you may encounter modifications. In this case, you need to follow the guide to modify and re-validate the qna.yaml.

#### Generating synthetic data with InsturctLab
RHEL AI generates synthetic data for LLM to train on based on qna.yaml.
This makes it possible to improve the quality of LLM learning with a relatively small number of qna.yamls.
Here is a command to generate synthetic data quickly by creating less synthetic data without a teacher model, taking into account the CPU environment.

```bash
$ ilab data generate --pipeline simple --sdg-scale-factor 5 --enable-serving-output 
```

> [!NOTE]
> The more synthetic data you generate, the better LLM training quality you can expect. For more information, see the [official documentation](https://docs.redhat.com/ko/documentation/red_hat_enterprise_linux_ai/1.2/html/creating_a_custom_llm_using_rhel_ai/generate_sdg).

> [!WARNING]
> The process takes about 30 minutes on an Intel(R) Core(TM) i7-14700 CPU.

### 4. Train for your LLM with InstuctLab
InstuctLab utilizes synthetic data generated to help train LLMs.
The following commands minimize training by taking into account the CPU environment.

```bash
(venv) $ ilab model train --pipeline simple --enable-serving-output
```

> [!WARNING]
> The process takes over 48 hours on an Intel(R) Core(TM) i7-14700 CPU. It is recommended to be performed in a GPU environment if possible.

### 5. Check your data training results
#### Changing Models in InstuctLab
Now let's verify that the trained LLM is what we want it to be.
First, restart the service by changing to the trained LLM.

```bash
(venv) $ ilab model serve --model-path ~/.local/share/instuctlab/checkpoints/ggml-model-f16.gguf
```

In chat_playground, let's query Zerobaseone again.

```
Tell me about K-pop artist Zerobaseone
```

Now you can see that the LLM alone is answering about this boy band.

<img width="100%" src="docs/imgs/trained_llm.jpg">

> [!NOTE]
> If the answer quality is low, try retraining by increasing the number of training data, increasing the epoch value, or increasing the number of iterations.

#### Filtering context in RHEL AI Pro Suite
Even with the learned LLM applied, the RAG is still sending tons of context with each query.
This is quite inefficient considering the network cost and the maximum number of tokens the LLM can handle.
Let's call the following API to filter the domain knowledge in the Vector DB.

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

Now, for the query, the Vector DB is searched, but RAG does not pass the Context to LLM because it thinks it has already learned what was searched for. You can see that LLM generates the answer faster.
You can also see that for the same query in chat_playground, RAG no longer sends the Context and generates the answer just fine.

<img width="100%" src="docs/imgs/context_filtered.jpg">


### 6. Validating models based on training data
A key feature of RHEL AI Pro Suite is the ability to validate models based on training data. While validation with common benchmarks is meaningful in terms of comparing the overall performance of a trained LLM to other LLMs, it is difficult to determine how well an LLM has been trained on the data it was trained on. However, RHEL AI Pro Suite can measure the search quality and answer quality of trained LLMs and RAG systems through various metrics.

First, serve the Critic LLM on another machine (for this example, using `mistral-7b-instruct-v0.2.Q4_K_M.gguf`).

> [!CAUTION]
> It is not recommended to use the Critic LLM on the same machine; run the Critic LLM on a different machine if possible.

> [!CAUTION]
> The Critic LLM recommends using large, high-performance models with a maximum number of tokens processed of 128K or more and a model size of 70B parameters or more. If you use a quantized model, it may deliver incorrect metric values due to low computational performance.

```bash
(venv) $ ilab model serve --model-path ~/.cache/instructlab/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

```
Afterwards, call the following APIs to validate the model based on the training data.

```bash
$ curl -X 'POST' \
  'http://localhost:8888/v1/qna/evaluate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "critic_llm": {
        "host": "http://localhost:8001",
        "headers": {
            "Content-Type": "application/json"
        }

    },
    "domain": "k-pop"
}'
```

After the API call, you can run the evaluation_dashboard to see various metrics.

<img width="100%" src="docs/imgs/evaluation.png">

> [!NOTE]
> Validation of RHEL AI Pro Suite is based on [RAGAS v0.1.21](https://docs.ragas.io/en/v0.1.21/), see [here](https://docs.ragas.io/en/v0.1.21/concepts/metrics/index.html) for usage metrics.

## API Endpoints
For information about the API, see [here](docs/rest_api.md).

## Configuration
For information about Configuration, see [here](docs/configuration.md).

## Demos and Slides
Demo videos and slides are available [here](docs/demos).

## License
RHEL AI Pro Suite is distributed under AGPL-3.0.
