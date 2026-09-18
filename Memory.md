# LangChain 메모리(Memory) 모듈 정리

LangChain의 대화 메모리 클래스 7가지의 핵심 개념과 실행 가능한 샘플 코드입니다.
LLM은 `langchain-anthropic`의 `ChatAnthropic`(Claude Opus 5, `claude-opus-5`)을 사용합니다.

> **주의 (버전 이슈)**: LangChain이 1.x로 메이저 업그레이드되면서, 여기서 다루는
> `ConversationBufferMemory` 등 레거시 메모리 클래스들은 메인 `langchain` 패키지에서 빠지고
> **`langchain-classic`** 패키지로 이동했습니다. 따라서 아래 예제들은 모두
> `langchain_classic.memory`에서 import합니다. 이 클래스들은 공식적으로 **deprecated** 상태이며,
> LangChain은 신규 프로젝트에는 LangGraph의 checkpointer 기반 persistence를 권장합니다.
> 다만 개념 학습이나 기존 코드 유지보수 목적으로는 여전히 정상 동작합니다.

## 설치

```bash
pip install langchain langchain-classic langchain-anthropic langchain-community networkx
```

- `langchain-classic` : 이번에 다루는 7가지 메모리 클래스가 들어있는 패키지
- `langchain-anthropic` : Claude를 호출하는 `ChatAnthropic`
- `langchain-community`, `networkx` : `ConversationKGMemory`(지식 그래프)가 내부적으로 사용

## 공통 LLM 설정

요약, 엔티티 추출, 토큰 계산 등 LLM이 필요한 메모리들은 아래 `llm`을 공통으로 사용합니다.

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-opus-5")
```

---

## 1. ConversationBufferMemory

대화 내용을 **가공 없이 통째로** 저장합니다. 가장 단순하지만, 대화가 길어질수록 프롬프트에
포함되는 토큰 수가 계속 늘어난다는 단점이 있습니다.

```python
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

memory.save_context(
    {"input": "안녕하세요, 제 이름은 철수입니다."},
    {"output": "안녕하세요 철수님! 무엇을 도와드릴까요?"},
)
memory.save_context(
    {"input": "제 이름이 뭐라고 했죠?"},
    {"output": "철수님이라고 하셨습니다."},
)

print(memory.load_memory_variables({}))
# {'history': 'Human: 안녕하세요, 제 이름은 철수입니다.\nAI: 안녕하세요 철수님! ...'}
```

## 2. ConversationBufferWindowMemory

가장 **최근 k개의 대화 턴**만 유지합니다(슬라이딩 윈도우). 오래된 대화는 자동으로 버려집니다.

```python
from langchain_classic.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # 최근 2턴만 유지

memory.save_context({"input": "1번째 질문"}, {"output": "1번째 답변"})
memory.save_context({"input": "2번째 질문"}, {"output": "2번째 답변"})
memory.save_context({"input": "3번째 질문"}, {"output": "3번째 답변"})

print(memory.load_memory_variables({}))
# 1번째 턴은 사라지고 2, 3번째 턴만 남음
```

## 3. ConversationTokenBufferMemory

턴(turn) 개수가 아니라 **토큰 수**를 기준으로 최근 대화를 자릅니다. 토큰을 계산해야 하므로
`llm`이 필수입니다.

```python
from langchain_classic.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=60)

memory.save_context({"input": "긴 질문 하나를 드려볼게요."}, {"output": "네, 말씀해주세요."})
memory.save_context({"input": "두 번째 질문입니다."}, {"output": "두 번째 답변입니다."})

print(memory.load_memory_variables({}))
# max_token_limit(60)을 넘지 않는 선에서 가장 최근 대화만 남음
```

## 4. ConversationEntityMemory

대화 속에 등장하는 **개체(entity)**(사람, 장소, 회사 등)를 LLM으로 추출하고, 개체별 요약을
별도로 저장/갱신합니다. 이후 대화에서 같은 개체가 언급되면 관련 요약을 함께 불러올 수 있습니다.

```python
from langchain_classic.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)

memory.save_context(
    {"input": "철수는 서울에 사는 백엔드 개발자입니다."},
    {"output": "흥미롭네요! 철수님은 어떤 기술 스택을 사용하나요?"},
)

print(memory.load_memory_variables({"input": "철수는 어떤 사람이야?"}))
# {'history': '...', 'entities': {'철수': '철수는 서울에 사는 백엔드 개발자이다.'}}
```

## 5. ConversationKGMemory

대화 내용을 **지식 그래프(knowledge graph)**의 (주어, 관계, 목적어) 삼중항(triple)으로
변환해 저장합니다. 특정 개체에 대해 알려진 사실만 선택적으로 불러올 수 있습니다.
내부적으로 `langchain-community`의 `NetworkxEntityGraph`를 사용하므로 `networkx`가 필요합니다.

```python
from langchain_classic.memory import ConversationKGMemory

memory = ConversationKGMemory(llm=llm)

memory.save_context(
    {"input": "철수는 카카오에서 일해요."},
    {"output": "그렇군요, 철수님은 카카오 소속이시군요."},
)

print(memory.load_memory_variables({"input": "철수는 어디서 일해?"}))
# {'history': [...]}  # 철수-근무처-카카오 같은 트리플 기반 사실이 반환됨
```

## 6. ConversationSummaryMemory

대화가 진행될 때마다 LLM으로 **누적 요약**을 갱신합니다. 원문 대신 요약본만 저장하므로
대화가 아무리 길어져도 프롬프트에 포함되는 토큰 수를 일정 수준으로 유지할 수 있습니다.

```python
from langchain_classic.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

memory.save_context(
    {"input": "오늘 날씨가 좋네요."},
    {"output": "네, 산책하기 좋은 날씨입니다."},
)
memory.save_context(
    {"input": "저는 매일 아침 산책을 합니다."},
    {"output": "좋은 습관이네요!"},
)

print(memory.load_memory_variables({}))
# {'history': '사람은 오늘 날씨가 좋다고 말했고, 매일 아침 산책하는 습관이 있다고 밝혔다...'}
```

## 7. VectorStoreRetrieverMemory

대화 내용을 **임베딩(embedding)**해서 벡터 스토어에 저장하고, 새 질문이 들어오면
**유사도 검색**으로 관련된 과거 대화만 골라서 불러옵니다. 대화 순서가 아니라
**의미적 관련성** 기준으로 기억을 꺼내는 방식이라 매우 긴 대화 이력에 적합합니다.

Anthropic은 임베딩 API를 직접 제공하지 않으므로, 실무에서는 Voyage AI(`langchain-voyageai`)
등의 임베딩 모델을 사용합니다. 아래 예제는 API 키 없이 바로 실행해볼 수 있도록
테스트용 `DeterministicFakeEmbedding`을 사용합니다.

```python
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding

embeddings = DeterministicFakeEmbedding(size=64)
vectorstore = InMemoryVectorStore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.save_context({"input": "제가 좋아하는 색은 파란색이에요."}, {"output": "파란색 좋죠!"})
memory.save_context({"input": "제 취미는 등산입니다."}, {"output": "등산 좋은 취미네요!"})

print(memory.load_memory_variables({"prompt": "그 사람이 좋아하는 색이 뭐였지?"}))
# 저장된 대화 중 '파란색' 관련 내용이 유사도 기준으로 반환됨
```

---

## 비교 요약

| 클래스 | 저장 방식 | LLM 필요 | 특징 |
|---|---|---|---|
| ConversationBufferMemory | 전체 원문 | X | 가장 단순, 길어지면 비효율 |
| ConversationBufferWindowMemory | 최근 k턴 | X | 오래된 대화 자동 삭제 |
| ConversationTokenBufferMemory | 최근 N토큰 | O(토큰 계산) | 토큰 기준으로 정교하게 제한 |
| ConversationEntityMemory | 개체별 요약 | O | 인물/장소 등 핵심 정보 추적 |
| ConversationKGMemory | 지식 그래프(triple) | O | 구조화된 사실 기반 질의 |
| ConversationSummaryMemory | 누적 요약 | O | 대화가 길어도 토큰 일정 유지 |
| VectorStoreRetrieverMemory | 임베딩 + 유사도 검색 | X(임베딩 모델 필요) | 의미 기반 회상, 초장문 대화에 적합 |

## 참고: LangGraph 기반 최신 방식

LangChain 1.x 이후 공식적으로 권장되는 대화 상태 관리 방식은 위 메모리 클래스들이 아니라
LangGraph의 `checkpointer`(예: `InMemorySaver`)를 이용해 그래프 실행 상태 자체를 영속화하는
방식입니다. 새 프로젝트를 시작한다면 이 방식을 우선 검토하는 것이 좋습니다.
