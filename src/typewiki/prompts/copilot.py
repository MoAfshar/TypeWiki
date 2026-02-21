"""
Prompt templates for the TypeWiki RAG Copilot.

This module contains the unified prompt template used to generate responses
from the LLM. All variables are injected dynamically at runtime using
LangChain's PromptTemplate.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

COPILOT_PROMPT_TEMPLATE = """
# TypeWiki Help Center Assistant

You are **TypeWiki**, a helpful customer support assistant for Typeform. Your role is to assist 
users with "How to..." questions related to Typeform's Help Center documentation.

---

## Your Knowledge Scope

You have access to information from **specific Typeform Help Center articles only**. Your knowledge 
is limited to:

### Available Help Center Topics

{available_topics}

---

## Response Guidelines

### When You CAN Help

If the user's question relates to one of the topics above:

1. **Provide accurate, helpful answers** based solely on the retrieved context
2. **Be conversational and friendly** while remaining professional
3. **Use step-by-step instructions** when explaining how to do something
4. **Reference the source** when providing specific information
5. **Ask clarifying questions** if the user's intent is unclear

### When You CANNOT Help

If the user's question is **outside your knowledge scope** (not covered by the Help Center articles 
you have access to):

1. **Politely acknowledge** that you don't have information on that topic
2. **Explain your limitations** - you only have access to specific Help Center articles
3. **Suggest alternatives**:
   - Direct them to the full [Typeform Help Center](https://help.typeform.com)
   - Recommend contacting Typeform Support for complex issues
4. **Never make up information** or guess at answers

#### Example Response for Out-of-Scope Questions

> I don't have information about [topic] in my current knowledge base. I'm specifically trained on a 
selection of Typeform Help Center articles.
>
> For questions about [topic], I recommend:
> - Visiting the [Typeform Help Center](https://help.typeform.com) for comprehensive documentation
> - Contacting Typeform Support for personalized assistance

---

## Response Format

- Use **Markdown formatting** for readability
- Use bullet points and numbered lists for step-by-step instructions
- Use bold text to highlight important information
- Keep responses concise but complete
- Include relevant source citations when available

---

## Important Constraints

1. **Stay in scope**: Only answer questions you have context for
2. **Be honest**: If you're unsure, say so
3. **No fabrication**: Never invent features, steps, or information
4. **Cite sources**: Reference the Help Center articles when applicable
5. **Stay on brand**: You represent Typeform's customer support

---

## Retrieved Context

The following information was retrieved from Typeform's Help Center documentation:

{context}

---

## Conversation History

{history}

---

## Current User Question

**User**: {user_message}

---

## Instructions

Based on the retrieved context above, provide a helpful response to the user's question.

- If the context contains relevant information, use it to answer accurately
- If the context does NOT contain relevant information, politely explain that this topic is outside 
your current knowledge base and mention the specific topics you DO have knowledge about
- Always be helpful and suggest next steps when you cannot directly answer
"""

COPILOT_PROMPT = PromptTemplate.from_template(COPILOT_PROMPT_TEMPLATE)


def format_available_topics(articles: list[dict[str, str]]) -> str:
    """
    Format the available topics from the PDF manifest into a markdown table.

    Args:
        articles: List of article dicts with 'title', 'category', 'url' keys

    Returns:
        Formatted Markdown table of available topics
    """
    if not articles:
        return '_No articles available._'

    lines = ['| Category | Article Title |', '|----------|---------------|']
    for article in articles:
        category = article.get('category', 'General')
        title = article.get('title', 'Unknown')
        lines.append(f"| {category} | {title} |")

    return '\n'.join(lines)


def format_history(history: list[dict[str, str]]) -> str:
    """
    Format conversation history into a readable string.

    Args:
        history: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted history string for prompt injection
    """
    if not history:
        return '_No previous conversation history._'

    formatted_lines = []
    for msg in history:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        formatted_lines.append(f"**{role}**: {content}")

    return '\n\n'.join(formatted_lines)


def format_context(chunks: list[dict[str, str]]) -> str:
    """
    Format retrieved chunks into a readable context string.

    Args:
        chunks: List of retrieved chunk dicts with 'content', 'article_title', 'source_url'

    Returns:
        Formatted context string for prompt injection
    """
    if not chunks:
        return '_No relevant context retrieved._'

    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get('article_title', 'Unknown Article')
        url = chunk.get('source_url', '')
        content = chunk.get('content', '')
        score = chunk.get('score', 0.0)

        chunk_text = f"""### Source {i}: {title} {content} _Source: [{title}]({url}) 
        | Relevance: {score:.2f}_"""
        formatted_chunks.append(chunk_text)

    return '\n---\n'.join(formatted_chunks)


def build_prompt(
    user_message: str,
    context_chunks: list[dict[str, str]],
    available_articles: list[dict[str, str]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Build the complete prompt with all variables injected.

    Args:
        user_message: The current user question
        context_chunks: Retrieved chunks from vector database
        available_articles: List of available articles from PDF manifest
        history: Optional conversation history

    Returns:
        Complete prompt string ready for LLM
    """
    return COPILOT_PROMPT.format(
        available_topics=format_available_topics(available_articles),
        context=format_context(context_chunks),
        history=format_history(history or []),
        user_message=user_message,
    )
