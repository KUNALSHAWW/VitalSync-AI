# 🏥 VitalSync AI: Intelligent Triage Assistant

<div align="center">

![VitalSync AI](assets/images/posts/README/im-778762.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Intelligent Medical Pre-Screening & Triage System - Bridging the gap between symptoms and care**

[Features](#features) • [Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage) • [Contributing](#contributing)

</div>

---

## About

**VitalSync AI** is an intelligent triage assistant that transforms how users understand and assess their medical symptoms. Built with advanced **LLM technologies**, **Retrieval Augmented Generation (RAG)**, and integrated **safety features**, VitalSync bridges the gap between symptom experience and professional medical care.

Unlike traditional chatbots, VitalSync AI includes:
- **🚨 Emergency Triage Layer** - Automatically detects critical situations and provides immediate emergency guidance
- **📄 Consultation Reports** - Generate downloadable PDF reports for your healthcare provider
- **🎯 Smart Symptom Analysis** - Context-aware responses using medical knowledge retrieval
- **🛡️ Safety-First Design** - Built-in disclaimers and professional medical advice recommendations

> **Important Disclaimer:** VitalSync AI is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. In case of emergency, call your local emergency services immediately.

### Release Information

- **Current Version:** 1.0.0
- **Release Date:** December 2025
- **Status:** Production-Ready

---

## Features

### 🚀 New VitalSync AI Features (v1.0.0)

- **🚨 Emergency Triage Layer**: 
  - Real-time detection of emergency keywords (heart attack, stroke, suicide ideation, etc.)
  - Automatic bypass of AI processing for critical situations
  - Immediate display of emergency contact numbers (911, 112, etc.)
  - Safety-first approach prioritizing user wellbeing

- **📄 Consultation Report Export**:
  - Professional PDF generation from chat history
  - Timestamped consultation transcripts
  - Medical-grade formatting with disclaimers
  - Easy sharing with healthcare providers

- **🎨 Professional Medical Dashboard**:
  - Clean teal/medical blue themed interface
  - Intuitive symptom input with guided prompts
  - Clear assessment display with follow-up recommendations
  - Prominent disclaimer and safety messaging

### Core Capabilities

- **Multi-Model Support**: Integration with advanced foundation models:
  - Meta Llama 3 fine-tuned for medical applications
  - Mixtral-8x7B for complex reasoning
  - Custom medical LLM wrapper for VitalSync responses

- **RAG-Powered Responses**: Retrieval Augmented Generation for accurate, context-aware medical information from curated medical knowledge base

- **Vector Database Integration**: Milvus for efficient similarity search across 1000+ medical dialogues

- **Interactive Interface**: Gradio-based web UI optimized for medical consultations

- **Production-Ready Infrastructure**:
  - Comprehensive test suite with pytest
  - Type hints and PEP 8 compliance
  - Makefile-driven development workflow
  - FPDF integration for report generation

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- UV package manager (recommended) or pip
- API keys for OpenAI and/or IBM WatsonX

### Installation

#### Using UV (Recommended)

```bash
# Install uv package manager
pip install uv

# Clone the repository
git clone https://github.com/KUNALSHAWW/VitalSync-AI.git
cd VitalSync-AI

# Install dependencies
make install

# For development
make install-dev

# For GPU support
make install-gpu
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/KUNALSHAWW/VitalSync-AI.git
cd VitalSync-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# IBM WatsonX Configuration (optional)
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_project_id_here

# Milvus Configuration (optional)
REMOTE_SERVER=127.0.0.1

# System Configuration
SYSTEM_MESSAGE=You are a helpful medical assistant.
```

---

## Usage

### Running VitalSync AI

```bash
# Using Makefile
make run-chatbot

# Or directly with Python
python 5-HuggingFace/app.py
```

The application will launch at `http://localhost:7860` with the VitalSync AI interface.

### Key Features in Action

1. **Symptom Analysis**: Type your symptoms in the "Describe Your Symptoms" box
2. **Emergency Detection**: Keywords like "chest pain" or "can't breathe" trigger immediate safety alerts
3. **Download Reports**: Click "📄 Download Report" to save your consultation as a PDF
4. **Safe Guidance**: All responses include disclaimers and recommendations to consult healthcare professionals

### Running the Medical Interviewer

```bash
# Using Makefile
make run-interviewer

# Or directly with Python
python 8-Interviewer/hf/app.py
```

### Using the Makefile

The project includes a comprehensive Makefile for common operations:

```bash
# Show all available commands
make help

# Code quality checks
make format          # Format code with black and isort
make lint            # Run linters (flake8, pylint)
make type-check      # Run mypy type checking
make check           # Run all quality checks

# Testing
make test            # Run all tests
make test-unit       # Run unit tests only
make test-integration # Run integration tests only
make test-cov        # Run tests with coverage report

# Cleaning
make clean           # Remove all artifacts
```

---

## Project Structure

The structure of the program contains the following main components:

1. [**Environment creation**](./1-Environment/README.md)

   Here we are going to create the environment to create the models locally that later can be used

2. [**Creation of the Medical Dataset.**](./2-Data/README.md)

   In this part we are going to build the Datasets that will be used create the **Medical Model**

3. [**Creation of the model by using RAG**](./3-Modeling/README.md)
   In this part we will perform feature engineering and create the model

4. [**Finetuning Models for the Medical Chatbot**](./6-FineTunning/README.md)
   We create a custom model based on medical information


5. [**Multimodal Medical Chatbot**](./7-Multimodal/README.md)
   We develop a medical chatbot multimodal, that from images can give you a description of the issue. We analazize different Medical Images Datasets.


## Chatbot with WatsonX

**Implementation of a chatbot with WatsonX in production.**

Here we will create a chatbot with the capability to answer questions by using the Model created before.
For Production in WatsonX you can checkout this repo


[Watsonx-Assistant-with-Milvus-as-Vector-Database](https://github.com/ruslanmv/Watsonx-Assistant-with-Milvus-as-Vector-Database)


## Chatbot with Custom LLM 
We have also developed another version which uses a custom LLM 

[Medical-Chatbot-with-Langchain-with-a-Custom-LLM](https://github.com/ruslanmv/Medical-Chatbot-with-Langchain-with-a-Custom-LLM)

## Playground Demo 


**VitalSync AI - Medical Chatbot by RAG method**.

[https://huggingface.co/spaces/KunalShaw/VitalSync-AI](https://huggingface.co/spaces/KunalShaw/VitalSync-AI)

[![](assets/images/posts/README/future.jpg)](https://huggingface.co/spaces/KunalShaw/VitalSync-AI)



**VitalSync AI - Medical Chatbot by using Medical-Llama3-8B**

[https://huggingface.co/spaces/KunalShaw/VitalSync-AI](https://huggingface.co/spaces/KunalShaw/VitalSync-AI)


[![](assets/2024-05-16-09-23-02.png)](https://huggingface.co/spaces/KunalShaw/VitalSync-AI)




## Fine-tunning Models with ai-medical chatbot

Currently there are two base models that were pretrained with ai-medical-chatbot

## Meta Llama 3
This repository provides a fine-tuned version of the powerful Llama3 8B model, specifically designed to answer medical questions in an informative way. It leverages the rich knowledge contained in the AI Medical Chatbot dataset.




[Medical-Llama3-8B](https://huggingface.co/ruslanmv/Medical-Llama3-8B)

The latest version of the Medical Llama 2 v2 with an improved Chatbot Interface in Google Colab


[Medical-Llama3-v2](https://huggingface.co/ruslanmv/Medical-Llama3-v2)



## Mixtral-7B
Fine-tuned Mixtral model for answering medical assistance questions. This model is a novel version of mistralai/Mistral-7B-Instruct-v0.2, adapted to a subset of 2.0k records from the AI Medical Chatbot dataset, which contains 250k records . The purpose of this model is to provide a ready chatbot to answer questions related to medical assistance.

[Medical-Mixtral-7B-v2k](https://huggingface.co/ruslanmv/Medical-Mixtral-7B-v2k)

For more details how was pretrained you can visit this post [here](https://ruslanmv.com/blog/How-to-Fine-Tune-Mixtral-87B-Instruct-model-with-PEFT)

> Let us use the best technologies in the world to help us. 



## Medical Interviewer
[![](assets/2024-09-08-19-33-56.png)](https://huggingface.co/spaces/ruslanmv/Medical-Interviewer)

Chatbot that perform medical interview

For more details visit [this](./8-Interviewer/README.md)


## DeepSeek-R1-Distill-Llama-8B

Currently we are developing  a new AI model in collaboration with the [Tilburg University](https://www.tilburguniversity.edu/), to create a new novel model able to understand your feelings.

The study of emotions and their underlying needs is a critical component of understanding human communication, particularly in contexts such as psychology, nonviolent communication (NVC), and conflict resolution. Emotional states often manifest as evaluative expressions—terms like "betrayed," "belittled," or "manipulated"—which not only convey subjective experiences but also point to unmet needs such as trust, respect, or autonomy. Effectively mapping these evaluative expressions to their associated feelings and corresponding needs is vital for creating tools that enhance emotional understanding and foster constructive dialogue.

[![image-20250203130739209](./assets/image-20250203130739209.png)](https://huggingface.co/spaces/ruslanmv/Empathy_Chatbot_v1)
You can test our current model [here](
https://huggingface.co/spaces/ruslanmv/Empathy_Chatbot_v1)

For more details of this project click [here](https://github.com/energycombined/empathyondemand)
## 🩺 Watsonx Medical MCP Server
Watsonx Medical MCP Server is a micro-service that wraps IBM watsonx.ai behind the MCP protocol, giving watsonx Orchestrate instant access to both a general-purpose chat endpoint (`chat_with_watsonx`) and a medical-symptom assessment tool (`analyze_medical_symptoms`).  


[![](https://github.com/ruslanmv/watsonx-medical-mcp-server/raw/master/docs/assets/2025-07-12-19-17-12.png)](https://github.com/ruslanmv/watsonx-medical-mcp-server/blob/master/docs/README.md)

Fully discoverable via STDIO, the server also exposes conversation-management helpers, rich resources/prompts, and ships with a Makefile-driven workflow for linting, auto-formatting, tests, and Docker packaging.  Zero-downtime reloads are achievable in development, and a lightweight Dockerfile plus CI workflow ensure smooth deployment. 

Explore the project [watsonx-medical-mcp-server](https://github.com/ruslanmv/watsonx-medical-mcp-server).

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Kunal Shaw

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Author

**Kunal Shaw**

- 📧 Email: kunalshawkol16@gmail.com
- 💼 GitHub: [KUNALSHAWW](https://github.com/KUNALSHAWW)
- 🤗 Hugging Face: [KunalShaw](https://huggingface.co/KunalShaw)

---

## Acknowledgments

- Original AI Medical Chatbot project by Ruslan Magana Vsevolodovna for the foundational architecture
- OpenAI for GPT models and APIs
- Hugging Face for model hosting and deployment infrastructure
- The open-source community for LangChain, Gradio, and other essential tools
- FPDF library for PDF generation capabilities

---

<div align="center">

**Made with ❤️ by [Kunal Shaw](https://github.com/KUNALSHAWW)**

[⬆ Back to Top](#-vitalsync-ai-intelligent-triage-assistant)

</div>





