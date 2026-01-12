# AI Agent with Amazon Bedrock AgentCore

This project demonstrates how to **build, deploy, and operate a production-ready AI agent**
The agent is designed to **answer customer questions** by retrieving relevant information from a **CSV-based FAQ knowledge base**, following an **Agentic RAG (Retrieval-Augmented Generation)** approach

The repository walks through the **entire lifecycle** of an AI agent — from **local prototyping** to a **fully managed, observable, and secure cloud deployment**.

---

## Project Overview

- **Goal**: Build a reliable AI agent that answers customer queries using a structured FAQ dataset
- **Approach**: Agentic RAG using LangChain + LangGraph
- **Deployment**: Docker-based deployment on Amazon Bedrock AgentCore
- **Key Focus**: Memory, observability, evaluation, security, and scalability

---

## Architecture at a Glance
```
User Query
↓
AI Agent (LangGraph Logic)
↓
Retriever (FAQ CSV → Embeddings)
↓
LLM Reasoning (Bedrock-hosted model)
↓
Memory + Policies + Tools (AgentCore)
↓
Final Answer

```
---

## Project Workflow

### Local Agent Development

The agent is first prototyped locally using modern open-source frameworks:

- **:contentReference[oaicite:1]{index=1}** – For RAG pipelines and tool abstraction  
- **:contentReference[oaicite:2]{index=2}** – For agentic control flow and state management  
- **Hugging Face Sentence Transformers** – For embedding FAQ data  
- **Groq + LLaMA models** – For fast local LLM inference  

**Dependency Management**
- Managed using **UV**, a modern alternative to `pip`
- Dependencies are defined and synchronized via `pyproject.toml`

---

### Infrastructure Configuration (AWS Setup)

To move from local to cloud:

- An **AWS account** is configured
- **AWS CLI** is installed for command-line access
- A **dedicated IAM user** is created with scoped permissions:
  - Amazon Bedrock
  - AgentCore
  - CodeBuild

This ensures **least-privilege security** while enabling full deployment capability

---

### Deployment to Amazon Bedrock AgentCore

The local agent code is adapted for AgentCore by:

- Importing `bedrock_agent_core_runtime`
- Adding an **entry-point function** for execution

**Deployment Steps**
1. `agentcore configure`  
   - Generates a YAML configuration file
2. `agentcore launch`  
   - Builds and deploys the agent as a **Docker-based container**

AgentCore abstracts away:
- Container orchestration
- Scaling
- Runtime infrastructure

## Conclusion

This project shows how **Amazon Bedrock AgentCore** removes the operational complexity traditionally associated with AI agents
