# Context Engineering for LLMs

> Reference documentation for token counting and context window management
> Sources: anthropic.com, winder.ai, oneuptime.com, agenta.ai

---

## 1. Token Counting with tiktoken

### 1.1 Basic Usage

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for a specific model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
```

### 1.2 Model-Encoding Mapping

| Model Family | Encoding | Notes |
|--------------|----------|-------|
| GPT-4, GPT-4o, GPT-3.5-turbo | cl100k_base | OpenAI standard |
| text-davinci-002/003 | p50k_base | Legacy |
| Claude 3 | claude | Use Anthropic's tokenizer |
| Llama 3 | llama | Use transformers tokenizer |
| Gemini | sentencepiece | Use Google's tokenizer |

### 1.3 Accurate Message Counting

For chat completion APIs, count the full message format:

```python
def count_chat_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """Count tokens in a chat completion request."""
    encoding = tiktoken.encoding_for_model(model)
    
    # OpenAI uses ~4 tokens per message for formatting
    tokens_per_message = 4
    
    total = 0
    for message in messages:
        total += tokens_per_message
        total += len(encoding.encode(message.get("role", "")))
        total += len(encoding.encode(message.get("content", "")))
    
    total += 3  # Reply priming
    return total
```

---

## 2. Context Window Limits

### 2.1 Current Model Limits (2025)

| Model | Context Window | Output Limit |
|-------|----------------|--------------|
| GPT-4o | 128K | 16K |
| GPT-4-turbo | 128K | 4K |
| Claude 3.5 Sonnet | 200K | 8K |
| Claude 3 Opus | 200K | 4K |
| Gemini 1.5 Pro | 2M | 8K |
| Llama 3 70B | 128K | 4K |

### 2.2 Golden Rule

```
input_tokens + output_tokens ≤ context_window
```

**Always reserve buffer for output!**

```python
def calculate_budget(
    max_context: int,
    expected_output: int,
    safety_margin: float = 0.1
) -> int:
    """Calculate input token budget."""
    reserved = expected_output + int(max_context * safety_margin)
    return max_context - reserved

# Example: 128K context, expect 4K output
budget = calculate_budget(128000, 4000)  # ~111,200 tokens
```

---

## 3. Context Optimization Strategies

### 3.1 Priority-Based Packing

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ContextItem:
    content: str
    priority: int  # Higher = more important
    tokens: int

def pack_context(items: List[ContextItem], budget: int) -> List[ContextItem]:
    """Pack highest priority items within budget."""
    sorted_items = sorted(items, key=lambda x: x.priority, reverse=True)
    
    packed = []
    used = 0
    
    for item in sorted_items:
        if used + item.tokens <= budget:
            packed.append(item)
            used += item.tokens
    
    return packed
```

### 3.2 Sliding Window for Conversations

```python
def sliding_window(
    messages: List[dict],
    budget: int,
    system_prompt: str,
    keep_first_n: int = 2,  # Keep first messages for context
) -> List[dict]:
    """Keep recent messages within budget, preserving system context."""
    
    system_tokens = count_tokens(system_prompt)
    first_messages = messages[:keep_first_n]
    first_tokens = sum(count_chat_tokens([m]) for m in first_messages)
    
    available = budget - system_tokens - first_tokens
    
    # Take most recent messages that fit
    recent = []
    for msg in reversed(messages[keep_first_n:]):
        msg_tokens = count_chat_tokens([msg])
        if available >= msg_tokens:
            recent.insert(0, msg)
            available -= msg_tokens
        else:
            break
    
    return [{"role": "system", "content": system_prompt}] + first_messages + recent
```

### 3.3 Summarization for Compression

```python
async def compress_history(
    messages: List[dict],
    llm_client,
    target_tokens: int = 500
) -> str:
    """Summarize conversation history for context."""
    
    full_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )
    
    summary = await llm_client.complete(
        f"Summarize this conversation in under {target_tokens} tokens, "
        f"preserving key facts and decisions:\n\n{full_text}"
    )
    
    return summary
```

### 3.4 RAG-Based Retrieval

Only include relevant context, not everything:

```python
async def retrieve_context(
    query: str,
    vector_store,
    budget: int,
    max_chunks: int = 10
) -> List[str]:
    """Retrieve only relevant context chunks."""
    
    hits = await vector_store.search(query, top_k=max_chunks)
    
    selected = []
    used = 0
    
    for hit in hits:
        tokens = count_tokens(hit.content)
        if used + tokens <= budget:
            selected.append(hit.content)
            used += tokens
    
    return selected
```

---

## 4. Anti-Patterns

### 4.1 Context Bloat

❌ **Bad**: Stuffing everything into context
```python
# Don't do this!
context = f"""
{entire_codebase}
{all_documentation}
{full_conversation_history}
{user_query}
"""
```

✅ **Good**: Curate relevant information
```python
context = f"""
{relevant_code_snippets}
{specific_docs}
{recent_conversation_summary}
{user_query}
"""
```

### 4.2 Ignoring Token Limits

❌ **Bad**: Hope it fits
```python
response = client.complete(messages=messages)  # May truncate or error
```

✅ **Good**: Validate before sending
```python
tokens = count_chat_tokens(messages)
if tokens > MAX_INPUT:
    messages = compress_to_fit(messages, MAX_INPUT)
response = client.complete(messages=messages)
```

### 4.3 Estimating Instead of Counting

❌ **Bad**: Character or word estimates
```python
estimated_tokens = len(text) / 4  # Unreliable!
```

✅ **Good**: Use actual tokenizer
```python
actual_tokens = count_tokens(text)
```

---

## 5. Performance Considerations

### 5.1 Token Counting Cost

tiktoken encoding has overhead. For high-throughput:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_count(text: str) -> int:
    """Cache token counts for repeated strings."""
    return count_tokens(text)
```

### 5.2 Lazy Counting

Only count when necessary:

```python
class LazyTokenItem:
    def __init__(self, content: str):
        self.content = content
        self._tokens = None
    
    @property
    def tokens(self) -> int:
        if self._tokens is None:
            self._tokens = count_tokens(self.content)
        return self._tokens
```

---

## 6. Cortex Implementation

The Context Engine in Cortex implements these patterns:

```python
# src/cortex/memory/context.py

def optimize_context(
    items: list[ContextItem],
    max_tokens: int,
    reserve_tokens: int = 0,
    model: str = "cl100k_base",
) -> PackingResult:
    """
    Optimize context items to fit within token budget.
    
    - Priority-based selection
    - Automatic token counting
    - Tracks included/dropped items
    """
    ...
```

**Gaps to Address**:
1. Add model-specific encoding selection
2. Implement summarization for dropped items
3. Add sliding window for conversation history
4. Consider content compression strategies

---

## References

- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://winder.ai/calculating-token-counts-llm-context-windows-practical-guide/
- https://oneuptime.com/blog/post/2026-01-30-context-window-management/view
- https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms
- https://eval.16x.engineer/blog/llm-context-management-guide
