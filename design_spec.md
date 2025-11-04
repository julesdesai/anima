# Self-Orchestrating User-Aligned Assistant
## Design Document v1.1 - Multi-Model Support

---

## Philosophy & Motivation

**Core Insight:** Replace weight fine-tuning with test-time compute and self-orchestration.

**The Bitter Lesson Applied:**
- Don't hand-code what's "factual" vs "stylistic" - let the model figure it out
- Don't design retrieval strategies - give tools and let the model discover strategies
- Don't build rigid pipelines - enable flexible, adaptive orchestration
- Maximize model agency, minimize human orchestration logic

**Competitive Position:**
- LoRA: Training compute → baked knowledge
- This system: Test-time compute → dynamic retrieval
- Recent evidence: Test-time scaling is more cost-effective

---

## System Architecture

### High-Level Flow

```
User Query
    ↓
[Agent (Claude/DeepSeek/Hermes) with Corpus Search Tool]
    ↓
    ├─→ Decides what to search
    ├─→ Makes multiple retrieval calls
    ├─→ Refines based on results
    └─→ Determines when sufficient
    ↓
Response (as user would write it)
```

### Core Components

```
┌─────────────────────────────────────┐
│   User Query                         │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   Agent Model (configurable)         │
│   - Claude Sonnet 4.5                │
│   - DeepSeek R1 / V3                 │
│   - Hermes 70B                       │
│                                      │
│   System: "You are modeled on User, │
│            use corpus tool freely"   │
└──────────────┬──────────────────────┘
               ↓
        [Tool Calls]
               ↓
┌─────────────────────────────────────┐
│   Vector Database                    │
│   - Raw user corpus                  │
│   - Minimal metadata                 │
│   - Semantic embeddings              │
└─────────────────────────────────────┘
```

---

## Style Transfer Mechanism

### How Style Transfer Works

The system ensures good style transfer through **retrieval-based in-context learning**:

1. **Agent retrieves style-relevant samples** based on query context
2. **Model sees actual user writing** in similar situations
3. **Model mimics style patterns** through in-context learning
4. **Optional verification loop** checks style consistency

### Core Style Transfer Strategy

```python
# What happens during a query:

1. Agent analyzes query context
   → "This needs a technical explanation in email format"

2. Agent searches for factual content
   → search_corpus("transformer architecture opinions")
   
3. Agent searches for style samples
   → search_corpus("technical email explanations")
   → Returns 3-5 examples of similar writing

4. Model generates response conditioned on:
   - System prompt: "You are modeled on [User]"
   - Factual context: Retrieved information
   - Style examples: Actual user writing samples
   
5. Model implicitly matches style through pattern mimicry
```

### Why This Beats LoRA for Style

**LoRA limitations:**
- Bakes single "average" style into weights
- Cannot handle context-specific style variation
- User writes differently in different contexts (formal email vs casual chat)
- Style evolves over time, weights are static

**Retrieval advantages:**
- **Context-aware style**: Retrieve email-style for emails, chat-style for chats
- **Dynamic adaptation**: New writing samples immediately available
- **Explicit examples**: Model sees exactly what style to match
- **Temporal awareness**: Can retrieve recent vs historical writing style
- **User variety**: Different formality levels, audiences, topics

**The key insight:** Users don't have a single "style" - they have style patterns that vary by context. Retrieval lets us match the right style for the right situation.

### Style Transfer Enhancements (Implementation Phases)

#### Phase 1: Basic Style Transfer (Implicit)
```python
# Current design - retrieval + in-context learning
# Agent searches corpus, retrieves relevant samples
# Model mimics style from examples
# No explicit verification
```

#### Phase 2: Style Verification Loop
```python
def respond_with_style_verification(self, query: str) -> dict:
    """Generate response with style consistency checking"""
    
    # Generate initial response
    response = self.agent.respond(query)
    
    # Compute style similarity
    style_score = compute_style_similarity(
        response["response"],
        user_corpus_samples=self.get_style_samples()
    )
    
    if style_score < STYLE_THRESHOLD:  # e.g., 0.75
        # Retry with stronger style emphasis
        retry_prompt = f"""Your previous response didn't match the user's style well.
        
        Previous: {response['response']}
        
        Rewrite to better match these examples:
        {style_samples}
        """
        response = self.agent.respond(retry_prompt)
    
    return response

def compute_style_similarity(text: str, user_samples: list[str]) -> float:
    """Measure style match using embedding similarity"""
    text_emb = embed(text)
    sample_embs = [embed(s) for s in user_samples]
    similarities = [cosine_sim(text_emb, s_emb) for s_emb in sample_embs]
    return np.mean(similarities)
```

#### Phase 3: Style-Aware Retrieval
```python
# Agent learns to retrieve different style types

def retrieve_style_samples(self, query: str, context: dict) -> list:
    """
    Retrieve style samples matching the query context
    
    For technical email: 
    - Search "technical explanations in email"
    - Filter by source="email", formality="high"
    
    For casual chat:
    - Search "casual conversation"
    - Filter by source="chat", formality="low"
    """
    
    # Agent decides query type
    query_type = self.classify_query(query)
    # Returns: {"medium": "email", "formality": "high", "domain": "technical"}
    
    # Construct style-specific search
    style_query = f"{query_type['domain']} {query_type['medium']} {query_type['formality']}"
    
    # Retrieve with filters
    samples = self.search_corpus(
        style_query,
        k=5,
        filters={
            "source": query_type['medium'],
            # More filters as needed
        }
    )
    
    return samples
```

#### Phase 4: Self-Verification Tools
```python
# Give agent tools to check its own style

tools = [
    {
        "name": "search_corpus",
        "description": "Search user's writing corpus",
        # ... parameters
    },
    {
        "name": "check_style_match",
        "description": "Verify if draft text matches user's style",
        "parameters": {
            "draft_text": {"type": "string"},
            "reference_type": {"type": "string", 
                              "enum": ["email", "chat", "document", "technical"]}
        }
    }
]

# Agent workflow:
# 1. Generate draft response
# 2. Call check_style_match(draft, "email")
# 3. Receive score: {"style_score": 0.72, "issues": ["too formal", "missing casual markers"]}
# 4. Revise draft based on feedback
# 5. Iterate until satisfied
```

### Style Metrics for Evaluation

```python
# eval/style_metrics.py

def evaluate_style_consistency(response: str, 
                               user_corpus: list[str]) -> dict:
    """
    Comprehensive style evaluation metrics
    """
    return {
        # Embedding-based similarity
        "embedding_similarity": compute_embedding_similarity(response, user_corpus),
        
        # Statistical style features
        "avg_sentence_length": compare_sentence_length(response, user_corpus),
        "vocabulary_overlap": compute_vocabulary_overlap(response, user_corpus),
        "formality_score": compare_formality(response, user_corpus),
        
        # LLM-as-judge
        "llm_style_rating": llm_judge_style_match(response, user_corpus),
        
        # Specific markers
        "punctuation_similarity": compare_punctuation_patterns(response, user_corpus),
        "paragraph_structure": compare_paragraph_structure(response, user_corpus),
    }

def llm_judge_style_match(response: str, user_samples: list[str]) -> float:
    """Use LLM to judge style similarity"""
    prompt = f"""
    User's writing samples:
    {user_samples}
    
    Generated response:
    {response}
    
    Rate how well the generated response matches the user's writing style (0-10).
    Consider: tone, formality, sentence structure, vocabulary, personality.
    Provide score and brief explanation.
    """
    judgment = llm_call(prompt)
    return extract_score(judgment)
```

---

## Model Selection & Comparison

### Supported Models

| Model | Provider | Strengths | Use Case | Cost/1M tokens |
|-------|----------|-----------|----------|----------------|
| **Claude Sonnet 4.5** | Anthropic | Best tool use, long context (200K), excellent instruction following | Primary recommendation, production use | ~$3 input / $15 output |
| **DeepSeek R1/V3** | DeepSeek | Extremely cost-effective, strong reasoning, good tool use | Cost-sensitive deployments, high-volume | ~$0.14 input / $0.28 output |
| **Hermes 70B** | Nous Research | Self-hostable, strong instruction following, uncensored | Privacy-critical, on-prem, full control | Self-hosted cost only |

### Model-Specific Characteristics

**Claude Sonnet 4.5:**
```python
Strengths:
✓ Most reliable tool use
✓ 200K context (handles large retrieval results)
✓ Excellent at self-orchestration
✓ Fast inference
✓ Strong style matching capabilities

Weaknesses:
✗ Most expensive option
✗ Requires API access (no self-hosting)
✗ Rate limits on free tier

Best for:
- Production deployments
- Users requiring highest quality
- Complex multi-step reasoning
- When latency matters
```

**DeepSeek R1 / V3:**
```python
Strengths:
✓ 20x cheaper than Claude
✓ Strong reasoning (R1 has chain-of-thought)
✓ Good tool use capabilities
✓ Large context (128K-200K depending on version)
✓ Handles technical content well

Weaknesses:
✗ Tool use slightly less reliable than Claude
✗ May need more iterations to reach same quality
✗ Smaller ecosystem/less documentation
✗ Potentially higher latency

Best for:
- Cost-sensitive applications
- High-volume deployments (1000+ queries/day)
- Technical/reasoning-heavy users
- Experimentation and development
```

**Hermes 70B:**
```python
Strengths:
✓ Fully self-hostable (complete privacy)
✓ No API rate limits
✓ Strong instruction following
✓ Uncensored (follows user intent closely)
✓ Good at long-form generation
✓ Fixed costs (hardware only)

Weaknesses:
✗ Requires GPU infrastructure (A100/H100)
✗ Tool use less mature than Claude/GPT
✗ Shorter context (32K-64K typical)
✗ May need more prompt engineering
✗ Higher initial setup cost

Best for:
- Privacy-critical applications (medical, legal)
- Enterprise on-prem deployments
- Users with existing GPU infrastructure
- No data leaving your infrastructure
```

### Model Selection Decision Tree

```
START: What's your priority?

├─ QUALITY & RELIABILITY
│  └─→ Claude Sonnet 4.5
│     Use when: Production, paying users, quality > cost
│
├─ COST EFFICIENCY
│  └─→ DeepSeek R1/V3
│     Use when: High volume, tight budgets, 20x cost matters
│
├─ PRIVACY & CONTROL
│  └─→ Hermes 70B
│     Use when: Data cannot leave infrastructure, have GPUs
│
└─ NOT SURE?
   └─→ Start with Claude, benchmark with DeepSeek, evaluate Hermes if needed
```

---

## Component Specifications

### 1. Vector Database

**Schema:**
```python
{
    "id": "uuid",
    "text": "raw text content (no preprocessing)",
    "metadata": {
        "timestamp": "ISO datetime",
        "source": "email | chat | document | code | note",
        "char_length": int
    },
    "embedding": [768-dim vector]
}
```

**No artificial categorization:**
- No "facts" vs "style" separation
- No formality tags
- No topic classification
- Just raw text + minimal context

**Embedding Strategy:**
- Use OpenAI `text-embedding-3-small` (fast, cheap, good enough)
- Chunk size: ~500-1000 characters (preserves context)
- Overlap: 100 characters (handles boundary cases)

**Storage Options:**
- **Local dev:** Qdrant (easy Docker setup)
- **Production:** Pinecone or Weaviate (managed)
- Start with Qdrant for prototyping

### 2. Search Tool

**Tool Definition:**
```python
{
    "name": "search_corpus",
    "description": "Search through the user's writing corpus using semantic similarity. Returns relevant excerpts that can help you understand what the user would know, think, or how they would express something.",
    "parameters": {
        "query": {
            "type": "string",
            "description": "Search query - be specific about what you're looking for"
        },
        "k": {
            "type": "integer", 
            "description": "Number of results to return (1-20)",
            "default": 5
        },
        "time_range": {
            "type": "object",
            "description": "Optional time filter",
            "properties": {
                "start": "ISO datetime or null",
                "end": "ISO datetime or null"
            }
        },
        "source_filter": {
            "type": "array",
            "description": "Optional filter by source type",
            "items": {"enum": ["email", "chat", "document", "code", "note"]}
        }
    },
    "required": ["query"]
}
```

**Tool Implementation:**
```python
def search_corpus(query: str, k: int = 5, time_range: dict = None, 
                  source_filter: list = None) -> list[dict]:
    """
    Execute semantic search against corpus.
    Returns: List of {text, metadata, similarity_score}
    """
    # Embed query
    query_embedding = embed(query)
    
    # Build filter
    filters = {}
    if time_range:
        filters["timestamp"] = time_range
    if source_filter:
        filters["source"] = {"$in": source_filter}
    
    # Search
    results = vector_db.search(
        vector=query_embedding,
        limit=k,
        filter=filters
    )
    
    return [{
        "text": r.payload["text"],
        "metadata": r.payload["metadata"],
        "similarity": r.score
    } for r in results]
```

### 3. Agent System (Multi-Model Support)

**System Prompt Template:**
```python
SYSTEM_PROMPT = """You are a digital assistant modeled on {user_name}'s communication style, knowledge, and judgment.

Your goal: Respond to queries as {user_name} would respond themselves.

You have access to {user_name}'s complete writing corpus through the search_corpus tool. Use it strategically:
- Search for relevant knowledge, opinions, or information about the topic
- Search for examples of how {user_name} communicates in similar contexts
- Make multiple searches if needed to build comprehensive understanding
- Use retrieved content to inform both WHAT you say and HOW you say it

You are not {user_name}, but a system trained to represent their perspective. Be transparent about this if asked.

Think step-by-step about what information you need and how to retrieve it."""
```

**Model-Specific System Prompt Adjustments:**

```python
# DeepSeek specific addition (leverage chain-of-thought)
DEEPSEEK_ADDITION = """
Before each search, briefly reason about what information would be most useful.
After retrieving results, evaluate if you need additional searches or if you have enough context.
"""

# Hermes specific addition (more explicit structure)
HERMES_ADDITION = """
Follow this process:
1. Analyze the query to understand what's needed
2. Plan your searches (what to look for, in what order)
3. Execute searches systematically
4. Synthesize findings into a response matching {user_name}'s style
"""
```

**Unified Agent Interface:**
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseAgent(ABC):
    """Abstract base class for all model agents"""
    
    def __init__(self, corpus_db, user_name: str):
        self.db = corpus_db
        self.user_name = user_name
        self.max_iterations = 20
        
    @abstractmethod
    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        """Model-specific API call"""
        pass
    
    @abstractmethod
    def _parse_tool_use(self, response: Any) -> List[Dict]:
        """Extract tool calls from model response"""
        pass
    
    def respond(self, query: str) -> Dict:
        """
        Main agent loop - model self-orchestrates retrieval.
        Same logic for all models, different implementations.
        """
        system_prompt = self._build_system_prompt()
        messages = [{"role": "user", "content": query}]
        tool_calls_log = []
        
        for iteration in range(self.max_iterations):
            response = self._call_model(system_prompt, messages)
            
            # Check if model is done
            if self._is_complete(response):
                return {
                    "response": self._extract_text(response),
                    "tool_calls": tool_calls_log,
                    "iterations": iteration + 1,
                    "model": self.__class__.__name__
                }
            
            # Execute tool calls
            tool_uses = self._parse_tool_use(response)
            if tool_uses:
                tool_results = []
                for tool_use in tool_uses:
                    result = self._execute_tool(tool_use)
                    tool_results.append(result)
                    tool_calls_log.append({
                        "tool": tool_use["name"],
                        "input": tool_use["input"],
                        "result_count": len(result) if isinstance(result, list) else 1
                    })
                
                # Add to conversation
                messages = self._update_messages(messages, response, tool_results)
        
        return {
            "response": "Max iterations reached",
            "tool_calls": tool_calls_log,
            "iterations": self.max_iterations,
            "model": self.__class__.__name__
        }
    
    def _execute_tool(self, tool_use: Dict) -> Any:
        """Execute search_corpus tool (same for all models)"""
        if tool_use["name"] == "search_corpus":
            return search_corpus(
                query=tool_use["input"]["query"],
                k=tool_use["input"].get("k", 5),
                time_range=tool_use["input"].get("time_range"),
                source_filter=tool_use["input"].get("source_filter")
            )
        return {"error": "Unknown tool"}
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build model-specific system prompt"""
        pass
```

**Claude Implementation:**
```python
import anthropic

class ClaudeAgent(BaseAgent):
    def __init__(self, corpus_db, user_name: str, 
                 model: str = "claude-sonnet-4.5-20250929"):
        super().__init__(corpus_db, user_name)
        self.client = anthropic.Anthropic()
        self.model = model
        
    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(user_name=self.user_name)
    
    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        return self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=[{
                "name": "search_corpus",
                "description": "Search user's writing corpus",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                        "time_range": {"type": "object"},
                        "source_filter": {"type": "array"}
                    },
                    "required": ["query"]
                }
            }],
            max_tokens=4096
        )
    
    def _is_complete(self, response: Any) -> bool:
        return response.stop_reason == "end_turn"
    
    def _parse_tool_use(self, response: Any) -> List[Dict]:
        tools = []
        for block in response.content:
            if block.type == "tool_use":
                tools.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        return tools
    
    def _extract_text(self, response: Any) -> str:
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""
    
    def _update_messages(self, messages: List[Dict], 
                        response: Any, tool_results: List) -> List[Dict]:
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool["id"],
                    "content": str(result)
                }
                for tool, result in zip(self._parse_tool_use(response), tool_results)
            ]
        })
        return messages
```

**DeepSeek Implementation:**
```python
from openai import OpenAI  # DeepSeek uses OpenAI-compatible API

class DeepSeekAgent(BaseAgent):
    def __init__(self, corpus_db, user_name: str,
                 model: str = "deepseek-reasoner"):
        super().__init__(corpus_db, user_name)
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model
        
    def _build_system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(user_name=self.user_name)
        return base + "\n\n" + DEEPSEEK_ADDITION
    
    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        # Add system message to messages for OpenAI-style API
        full_messages = [{"role": "system", "content": system}] + messages
        
        return self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "description": "Search user's writing corpus",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "k": {"type": "integer", "default": 5},
                            "time_range": {"type": "object"},
                            "source_filter": {"type": "array"}
                        },
                        "required": ["query"]
                    }
                }
            }],
            temperature=1.0
        )
    
    def _is_complete(self, response: Any) -> bool:
        return (response.choices[0].finish_reason == "stop" or 
                response.choices[0].finish_reason == "end_turn")
    
    def _parse_tool_use(self, response: Any) -> List[Dict]:
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        
        tools = []
        for tool_call in message.tool_calls:
            tools.append({
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": json.loads(tool_call.function.arguments)
            })
        return tools
    
    def _extract_text(self, response: Any) -> str:
        return response.choices[0].message.content or ""
    
    def _update_messages(self, messages: List[Dict], 
                        response: Any, tool_results: List) -> List[Dict]:
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls
        })
        
        # Add tool results
        for tool_call, result in zip(response.choices[0].message.tool_calls, tool_results):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
        return messages
```

**Hermes Implementation:**
```python
from openai import OpenAI  # For vLLM/text-generation-inference compatibility

class HermesAgent(BaseAgent):
    def __init__(self, corpus_db, user_name: str,
                 base_url: str = "http://localhost:8000/v1",
                 model: str = "NousResearch/Hermes-2-Pro-Llama-3-70B"):
        super().__init__(corpus_db, user_name)
        self.client = OpenAI(
            api_key="not-needed",  # Local inference
            base_url=base_url
        )
        self.model = model
        self.max_iterations = 15  # May need fewer for context limits
        
    def _build_system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(user_name=self.user_name)
        return base + "\n\n" + HERMES_ADDITION
    
    def _call_model(self, system: str, messages: List[Dict]) -> Any:
        # Hermes expects very explicit tool formatting
        full_messages = [{"role": "system", "content": system}] + messages
        
        return self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "description": "Search through the user's writing corpus",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            }],
            temperature=0.7,
            max_tokens=2048  # Lower for local inference
        )
    
    def _is_complete(self, response: Any) -> bool:
        return response.choices[0].finish_reason in ["stop", "end_turn"]
    
    def _parse_tool_use(self, response: Any) -> List[Dict]:
        message = response.choices[0].message
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return []
        
        tools = []
        for tool_call in message.tool_calls:
            try:
                tools.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)
                })
            except:
                # Hermes sometimes outputs malformed JSON
                continue
        return tools
    
    def _extract_text(self, response: Any) -> str:
        return response.choices[0].message.content or ""
    
    def _update_messages(self, messages: List[Dict], 
                        response: Any, tool_results: List) -> List[Dict]:
        # Similar to DeepSeek but more defensive
        if response.choices[0].message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls
            })
            
            for tool_call, result in zip(response.choices[0].message.tool_calls, tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        return messages
```

**Agent Factory:**
```python
class AgentFactory:
    """Factory for creating appropriate agent based on config"""
    
    @staticmethod
    def create(model_name: str, corpus_db, user_name: str) -> BaseAgent:
        if model_name.startswith("claude"):
            return ClaudeAgent(corpus_db, user_name, model_name)
        elif model_name.startswith("deepseek"):
            return DeepSeekAgent(corpus_db, user_name, model_name)
        elif "hermes" in model_name.lower():
            return HermesAgent(corpus_db, user_name, model=model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

# Usage
agent = AgentFactory.create("claude-sonnet-4.5-20250929", corpus_db, "John Doe")
response = agent.respond("What do I think about AI safety?")
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Goal:** Get basic system working end-to-end with all three models

**Tasks:**
1. **Corpus Ingestion Pipeline**
   ```python
   # corpus_ingest.py
   - Load user text files (txt, md, emails, etc.)
   - Chunk into semantic units (~500-1000 chars)
   - Generate embeddings
   - Store in Qdrant with metadata
   ```

2. **Vector DB Setup**
   ```python
   # vector_db.py
   - Docker compose for Qdrant
   - Collection schema
   - Search implementation
   - Index optimization
   ```

3. **Multi-Model Agent Framework**
   ```python
   # agent/
   - base.py: BaseAgent abstract class
   - claude_agent.py: Claude implementation
   - deepseek_agent.py: DeepSeek implementation
   - hermes_agent.py: Hermes implementation
   - factory.py: Agent factory
   ```

**Deliverable:** Can query system with any of the three models

### Phase 2: Evaluation & Iteration (Week 2)

**Goal:** Compare model performance and optimize

**Tasks:**
1. **Evaluation Harness**
   ```python
   # eval.py
   - Test queries (factual, opinion, stylistic)
   - Ground truth from corpus
   - Run same queries across all models
   - Compare results
   ```

2. **Model-Specific Metrics**
   ```python
   # metrics.py
   - Style consistency (embedding distance)
   - Factual accuracy (retrieval quality)
   - Retrieval efficiency (# of tool calls)
   - Cost per query
   - Latency
   - Success rate (% completing without errors)
   ```

3. **Prompt Tuning Per Model**
   - Iterate on model-specific system prompts
   - Add examples if needed (especially for Hermes)
   - Tune retrieval parameters

4. **Style Verification (Optional)**
   ```python
   # style_verification.py
   - Implement style similarity metrics
   - Add verification loop if needed
   - Test impact on quality
   ```

**Deliverable:** Performance comparison matrix across models

### Phase 3: Production Features (Week 3)

**Goal:** Make it robust and usable

**Tasks:**
1. **Model Selection Logic**
   ```python
   # model_router.py
   - Auto-select based on query complexity
   - Fallback handling (if one model fails, try another)
   - Cost-based routing
   - Latency-based routing
   ```

2. **Conversation History**
   ```python
   # conversation.py
   - Multi-turn support
   - Context management (model-specific limits)
   - History pruning strategies
   ```

3. **API & Interface**
   ```python
   # api.py
   - FastAPI endpoint
   - Model selection parameter
   - Streaming responses
   - Tool call visibility
   - Cost tracking
   - Simple web UI with model selector
   ```

**Deliverable:** Deployable system with model choice

### Phase 4: Advanced Features (Week 4+)

**Goal:** Optimize for production

**Tasks:**
1. **Hermes Self-Hosting Setup**
   - vLLM deployment guide
   - GPU requirements
   - Performance benchmarks
   
2. **Hybrid Model Strategy**
   - Use DeepSeek for classification/retrieval
   - Use Claude for final synthesis
   - Cost optimization strategies

3. **Model Ensemble**
   - Multi-model consensus for critical queries
   - Confidence scoring
   - Best-of-N sampling

4. **Advanced Style Transfer**
   - Style verification loop
   - Self-verification tools
   - Style-aware retrieval strategies

---

## Configuration

**config.yaml:**
```yaml
# User configuration
user:
  name: "User Name"
  corpus_path: "data/corpus/"

# Model configuration - PRIMARY
model:
  primary: "claude-sonnet-4.5-20250929"
  fallback: "deepseek-reasoner"
  
  # Model-specific configs
  claude:
    api_key_env: "ANTHROPIC_API_KEY"
    max_tokens: 4096
    temperature: 1.0
    max_iterations: 20
    
  deepseek:
    api_key_env: "DEEPSEEK_API_KEY"
    base_url: "https://api.deepseek.com"
    model: "deepseek-reasoner"  # or "deepseek-chat"
    max_tokens: 4096
    temperature: 1.0
    max_iterations: 20
    
  hermes:
    base_url: "http://localhost:8000/v1"  # vLLM endpoint
    model: "NousResearch/Hermes-2-Pro-Llama-3-70B"
    max_tokens: 2048  # Conservative for local
    temperature: 0.7
    max_iterations: 15  # Fewer due to context limits

# Agent configuration
agent:
  max_tool_calls_per_iteration: 3
  system_prompt_dir: "prompts/"

# Vector DB configuration
vector_db:
  provider: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "user_corpus"
  
# Embedding configuration
embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
  batch_size: 100

# Corpus processing
corpus:
  chunk_size: 800
  chunk_overlap: 100
  min_chunk_length: 100
  file_types: [".txt", ".md", ".email", ".json"]

# Retrieval configuration
retrieval:
  default_k: 5
  max_k: 20
  similarity_threshold: 0.7

# Style verification (optional)
style:
  verification_enabled: false
  similarity_threshold: 0.75
  verification_method: "embedding"  # or "llm_judge"

# Cost tracking
cost_tracking:
  enabled: true
  log_path: "logs/costs.json"
  budget_alert_threshold: 10.0  # USD per day
```

---

## Project Structure

```
user-aligned-assistant/
├── README.md
├── requirements.txt
├── docker-compose.yml          # Qdrant + optional vLLM
├── .env.example               # API keys template
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── corpus/
│   │   ├── ingest.py          # Load and chunk corpus
│   │   ├── embed.py           # Embedding generation
│   │   └── update.py          # Incremental updates
│   ├── database/
│   │   ├── vector_db.py       # Qdrant interface
│   │   └── schema.py          # Data models
│   ├── agent/
│   │   ├── base.py            # BaseAgent abstract class
│   │   ├── claude_agent.py    # Claude implementation
│   │   ├── deepseek_agent.py  # DeepSeek implementation
│   │   ├── hermes_agent.py    # Hermes implementation
│   │   ├── factory.py         # Agent factory
│   │   ├── tools.py           # Tool definitions
│   │   └── prompts/           # Model-specific prompts
│   │       ├── base.txt
│   │       ├── deepseek.txt
│   │       └── hermes.txt
│   ├── eval/
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── harness.py         # Test runner
│   │   ├── model_comparison.py # Cross-model analysis
│   │   ├── style_metrics.py   # Style consistency metrics
│   │   └── test_queries.json  # Test cases
│   ├── routing/
│   │   ├── router.py          # Model selection logic
│   │   └── cost_optimizer.py  # Cost-based routing
│   └── api/
│       ├── server.py          # FastAPI app
│       ├── models.py          # API schemas
│       └── cost_tracker.py    # Usage tracking
│
├── scripts/
│   ├── setup_db.py            # Initialize vector DB
│   ├── ingest_corpus.sh       # Run ingestion
│   ├── run_eval.py            # Run evaluation
│   ├── compare_models.py      # Model comparison script
│   └── deploy_hermes.sh       # Deploy local Hermes
│
├── data/
│   ├── corpus/                # User text files
│   │   ├── emails/
│   │   ├── chats/
│   │   └── documents/
│   └── eval/
│       └── test_cases.json
│
├── notebooks/
│   ├── model_exploration.ipynb    # Compare models
│   └── cost_analysis.ipynb        # Cost optimization
│
├── deployment/
│   ├── hermes/
│   │   ├── vllm_config.yaml   # vLLM configuration
│   │   └── gpu_requirements.md
│   └── docker/
│       ├── Dockerfile.api
│       └── Dockerfile.hermes
│
└── tests/
    ├── test_agent.py
    ├── test_corpus.py
    ├── test_vector_db.py
    └── test_all_models.py
```

---

## Model-Specific Deployment Guides

### Claude Deployment

**Requirements:**
- Anthropic API key
- Internet connection

**Setup:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
```

**Cost Calculation:**
```python
# Per query estimate (assuming 5 tool calls)
# Input: ~2K tokens (system + query + 5 retrievals) = $0.006
# Output: ~500 tokens = $0.0075
# Total: ~$0.014 per query
# 1000 queries/day = $14/day
```

### DeepSeek Deployment

**Requirements:**
- DeepSeek API key
- Internet connection

**Setup:**
```bash
export DEEPSEEK_API_KEY="sk-..."
pip install openai  # DeepSeek uses OpenAI SDK
```

**Cost Calculation:**
```python
# Per query estimate (assuming 5 tool calls)
# Input: ~2K tokens = $0.00028
# Output: ~500 tokens = $0.00014
# Total: ~$0.00042 per query (33x cheaper than Claude!)
# 1000 queries/day = $0.42/day
```

**Note:** DeepSeek R1 (reasoner) is slightly more expensive but shows better reasoning chains.

### Hermes 70B Self-Hosted Deployment

**Requirements:**
- 2x A100 (80GB) or 4x A100 (40GB) for full precision
- 1x A100 (80GB) for int4 quantization
- H100 for optimal performance

**Setup with vLLM:**
```bash
# Install vLLM
pip install vllm

# Start server (int4 quantization for single A100)
python -m vllm.entrypoints.openai.api_server \
    --model NousResearch/Hermes-2-Pro-Llama-3-70B \
    --quantization awq \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --port 8000

# Or use Docker
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model NousResearch/Hermes-2-Pro-Llama-3-70B \
    --quantization awq
```

**Cost Calculation:**
```python
# Fixed costs (per month):
# 1x A100 (80GB): ~$1000-1500/month (cloud) or amortized hardware cost
# Electricity: ~$100-200/month
# 
# Per query: $0 (already paid for infrastructure)
# Break-even: ~50-100 queries/day vs Claude
# Break-even: ~1000-2000 queries/day vs DeepSeek
# 
# Sweet spot: >1000 queries/day + privacy requirements
```

**Docker Compose:**
```yaml
# docker-compose.hermes.yml
version: '3.8'
services:
  hermes:
    image: vllm/vllm-openai:latest
    command: >
      --model NousResearch/Hermes-2-Pro-Llama-3-70B
      --quantization awq
      --tensor-parallel-size 1
      --max-model-len 4096
      --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Evaluation Strategy

### Cross-Model Test Suite

```python
# eval/test_queries.json
{
  "factual_recall": [
    {"query": "What's my dog's name?", "expected": "Max"},
    {"query": "Where did I go to school?", "expected": "MIT"}
  ],
  "opinion_judgment": [
    {"query": "What do I think about test-time compute?"},
    {"query": "How do I feel about LoRA fine-tuning?"}
  ],
  "stylistic": [
    {"query": "Write an email to my advisor about delays"},
    {"query": "Explain quantum mechanics casually"}
  ],
  "complex_reasoning": [
    {"query": "Based on my research interests, what should I work on next?"},
    {"query": "How would I approach evaluating AI alignment?"}
  ]
}
```

### Model Comparison Metrics

```python
# eval/model_comparison.py

class ModelComparison:
    """Compare all models on same test suite"""
    
    def __init__(self, test_queries: dict, corpus_db):
        self.queries = test_queries
        self.db = corpus_db
        self.models = [
            ("Claude Sonnet 4.5", ClaudeAgent(corpus_db, "Test User")),
            ("DeepSeek R1", DeepSeekAgent(corpus_db, "Test User")),
            ("Hermes 70B", HermesAgent(corpus_db, "Test User"))
        ]
    
    def run_comparison(self) -> pd.DataFrame:
        results = []
        
        for category, queries in self.queries.items():
            for query_data in queries:
                query = query_data["query"]
                
                for model_name, agent in self.models:
                    start = time.time()
                    response = agent.respond(query)
                    latency = time.time() - start
                    
                    results.append({
                        "category": category,
                        "query": query,
                        "model": model_name,
                        "response": response["response"],
                        "tool_calls": len(response["tool_calls"]),
                        "iterations": response["iterations"],
                        "latency": latency,
                        "cost": self._estimate_cost(model_name, response)
                    })
        
        return pd.DataFrame(results)
    
    def _estimate_cost(self, model: str, response: dict) -> float:
        """Estimate cost per query"""
        if "Claude" in model:
            return 0.014
        elif "DeepSeek" in model:
            return 0.00042
        elif "Hermes" in model:
            return 0.0  # Self-hosted
        return 0.0

# Run comparison
comparison = ModelComparison(test_queries, corpus_db)
df = comparison.run_comparison()

# Analysis
print("Average Latency by Model:")
print(df.groupby("model")["latency"].mean())

print("\nAverage Tool Calls by Model:")
print(df.groupby("model")["tool_calls"].mean())

print("\nTotal Cost for 1000 queries:")
print(df.groupby("model")["cost"].sum() * 1000)
```

### Metrics Per Model

```python
# Expected performance characteristics

CLAUDE_SONNET_45 = {
    "avg_latency": 3.5,  # seconds
    "tool_call_success_rate": 0.98,
    "style_consistency": 0.85,
    "factual_accuracy": 0.92,
    "cost_per_1k_queries": 14.0,  # USD
    "context_window": 200_000
}

DEEPSEEK_R1 = {
    "avg_latency": 5.0,  # slightly slower
    "tool_call_success_rate": 0.90,  # less reliable
    "style_consistency": 0.80,  # good but not best
    "factual_accuracy": 0.88,
    "cost_per_1k_queries": 0.42,  # USD (33x cheaper)
    "context_window": 128_000
}

HERMES_70B = {
    "avg_latency": 2.5,  # fastest (local)
    "tool_call_success_rate": 0.85,  # needs more prompt engineering
    "style_consistency": 0.78,
    "factual_accuracy": 0.85,
    "cost_per_1k_queries": 0.0,  # fixed infrastructure
    "context_window": 32_000  # smallest
}
```

---

## Cost Analysis & Recommendations

### Scenario-Based Model Selection

**Scenario 1: Personal Use (< 100 queries/day)**
```
Recommendation: Claude Sonnet 4.5
- Cost: ~$1.40/day
- Best quality for personal assistant
- Worth the premium for < 100 queries
```

**Scenario 2: Small Business (100-1000 queries/day)**
```
Recommendation: DeepSeek R1 with Claude fallback
- Cost: ~$0.42-4.20/day (DeepSeek) + $1-2/day (Claude for 10%)
- Total: ~$2-6/day
- 80% savings vs full Claude
```

**Scenario 3: Enterprise (>1000 queries/day)**
```
Recommendation: Hermes 70B self-hosted
- Fixed cost: ~$1500/month infrastructure
- Variable cost: $0
- Break-even at ~3500 queries/day vs DeepSeek
- Full data control + privacy
```

**Scenario 4: Privacy-Critical (medical, legal, financial)**
```
Recommendation: Hermes 70B self-hosted (required)
- No data leaves infrastructure
- Full audit trail
- Compliance with HIPAA/GDPR/SOC2
```

---

## Hermes-Specific Considerations

### Tool Use Challenges

Hermes 70B has less mature tool use than Claude/GPT. Mitigations:

1. **More Explicit Prompting:**
```python
HERMES_TOOL_EMPHASIS = """
CRITICAL: You MUST use the search_corpus tool.

Process:
1. Think about what information you need
2. Call search_corpus with a specific query
3. Review the results
4. If needed, call search_corpus again with refined query
5. Once you have enough information, provide your response

Do NOT attempt to answer without searching the corpus first.
"""
```

2. **Few-Shot Examples:**
```python
HERMES_FEW_SHOT = """
Example 1:
User: What's my opinion on AI safety?
Assistant: <thinks> I need to search for the user's opinions on AI safety.
<tool_call>search_corpus("AI safety opinions")</tool_call>
[results returned]
Based on your writings, you believe...

Example 2:
User: Write an email to my advisor
Assistant: <thinks> I need to find how the user typically writes emails.
<tool_call>search_corpus("emails to advisor")</tool_call>
[results returned]
Here's a draft in your style...
"""
```

3. **Validation Layer:**
```python
class HermesAgent(BaseAgent):
    def respond(self, query: str) -> dict:
        response = super().respond(query)
        
        # Verify tool use happened
        if len(response["tool_calls"]) == 0:
            # Force retry with stronger prompt
            retry_prompt = f"""
            CRITICAL ERROR: You did not use the search_corpus tool.
            You MUST search the corpus before responding.
            
            Original query: {query}
            
            Now search the corpus and try again.
            """
            return super().respond(retry_prompt)
        
        return response
```

### Context Window Management

Hermes typically has 32K context vs Claude's 200K:

```python
class HermesAgent(BaseAgent):
    def _truncate_retrieval_results(self, results: list, 
                                   max_tokens: int = 8000) -> list:
        """
        Hermes has smaller context - truncate retrieved results
        """
        total_tokens = 0
        truncated = []
        
        for result in results:
            result_tokens = len(result["text"]) // 4  # rough estimate
            if total_tokens + result_tokens > max_tokens:
                break
            truncated.append(result)
            total_tokens += result_tokens
        
        return truncated
```

---

## Next Steps: Implementation with Claude Code

### Step 1: Environment Setup
```bash
# Initialize project
mkdir user-aligned-assistant && cd user-aligned-assistant
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install anthropic openai qdrant-client python-dotenv pydantic fastapi

# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...  # For embeddings
EOF
```

### Step 2: Start with Corpus Ingestion
```python
# scripts/ingest_corpus.py - First script to write
# Takes a directory of text files, chunks them, embeds them, stores in Qdrant
```

### Step 3: Implement Base Agent + One Model
```python
# Start with Claude (most reliable tool use)
# src/agent/base.py - Abstract interface
# src/agent/claude_agent.py - First concrete implementation
```

### Step 4: Add DeepSeek
```python
# src/agent/deepseek_agent.py - Second implementation
# Compare performance with evaluation harness
```

### Step 5: Add Hermes (Optional)
```python
# src/agent/hermes_agent.py - Only if self-hosting needed
# Requires GPU setup
```

### Step 6: Simple CLI
```python
# scripts/chat.py - Test interface
# python chat.py --model claude "What do I think about X?"
# python chat.py --model deepseek "What do I think about X?"
```

### Step 7: Evaluation & Comparison
```python
# scripts/compare_models.py
# Run same test suite on all models, generate comparison report
```

---

## Success Metrics (6 Month Vision)

**Technical:**
- Response quality ≥ LoRA fine-tuned baseline (all models)
- Claude: < 5s latency
- DeepSeek: < 7s latency  
- Hermes: < 3s latency
- Support 100K+ token corpus per user
- Style consistency > 0.75 (all models)

**Cost Efficiency:**
- 90% cost reduction vs Claude-only (with DeepSeek)
- Break-even on Hermes at 3000+ queries/day
- Actual cost per interaction < $0.01 average

**User Experience:**
- Users can't distinguish from their own writing (blind test)
- Handles novel topics user hasn't written about
- Maintains consistency across conversation
- Transparent about retrieval sources

**Business:**
- Scales to 1000+ users
- Zero training cost per user
- Instant onboarding (just upload corpus)
- Model-agnostic (easy to add GPT-5, Claude 5, etc.)

---

## Risk Mitigation

**Risk:** Model hallucinates despite corpus access
- **Mitigation:** LLM-as-judge verification, cite sources, user feedback loop

**Risk:** Poor retrieval → wrong responses  
- **Mitigation:** Log retrieval quality, add reranking, tune embeddings

**Risk:** Too slow (many tool calls)
- **Mitigation:** Caching, parallel search, tune iteration budget

**Risk:** Cost too high for production
- **Mitigation:** Use DeepSeek for 80% of queries, Claude for complex 20%

**Risk:** Style inconsistency
- **Mitigation:** Implement style verification loop, tune retrieval strategies

**Risk:** Privacy concerns with stored corpus
- **Mitigation:** Encryption at rest, user controls, audit logs, Hermes for sensitive data

---

## Appendix: Quick Start with Each Model

### Claude Quick Start
```python
from anthropic import Anthropic
from agent.claude_agent import ClaudeAgent

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
agent = ClaudeAgent(corpus_db, user_name="John Doe")
response = agent.respond("What do I think about AI?")
print(response["response"])
```

### DeepSeek Quick Start
```python
from openai import OpenAI
from agent.deepseek_agent import DeepSeekAgent

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
agent = DeepSeekAgent(corpus_db, user_name="John Doe")
response = agent.respond("What do I think about AI?")
print(response["response"])
```

### Hermes Quick Start
```bash
# Start vLLM server first
docker-compose -f docker-compose.hermes.yml up -d

# Then in Python
from openai import OpenAI
from agent.hermes_agent import HermesAgent

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)
agent = HermesAgent(corpus_db, user_name="John Doe")
response = agent.respond("What do I think about AI?")
print(response["response"])
```

---

## Minimal Working Example

```python
# main.py - 100 lines to working system

import anthropic
from qdrant_client import QdrantClient
import openai

# Setup
qdrant = QdrantClient(host="localhost", port=6333)
anthropic_client = anthropic.Anthropic()

def search_corpus(query: str, k: int = 5):
    # Embed query
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    # Search Qdrant
    results = qdrant.search(
        collection_name="user_corpus",
        query_vector=emb,
        limit=k
    )
    
    return [{"text": r.payload["text"], "score": r.score} for r in results]

def agent_respond(query: str, user_name: str):
    system = f"""You are a digital assistant modeled on {user_name}.
    Use the search_corpus tool to find relevant information from their writing."""
    
    messages = [{"role": "user", "content": query}]
    tools = [{
        "name": "search_corpus",
        "description": "Search user's writing corpus",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }]
    
    for _ in range(10):  # Max iterations
        response = anthropic_client.messages.create(
            model="claude-sonnet-4.5-20250929",
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=4096
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        # Execute tools
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = search_corpus(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached"

# Use it
response = agent_respond("What do I think about AI safety?", "John Doe")
print(response)
```

---

**Ready to build with Claude Code?** Start with corpus ingestion, then Claude agent, then add DeepSeek for cost comparison, then optionally Hermes if you have GPUs.