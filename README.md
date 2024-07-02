
# Text Summarization Using AutoGen

This project demonstrates text summarization using multiple AI agents with AutoGen. It leverages local models, including `facebook/bart-large-cnn` and `t5-base`, to generate concise summaries of input text.

## Table of Contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Setup](#setup)
* [Usage](#usage)
* [Classes and Methods](#classes-and-methods)
* [Logic](#logic)
* [Expected Outcome](#expected-outcome)
* [Example Output](#example-output)

## Introduction

This script showcases how to use local models and the AutoGen library to create multiple agents for text summarization. It demonstrates the integration of the BART and T5 models to produce summaries for long text inputs.

## Requirements

* Python 3.7 or higher
* `transformers` library
* `autogen` library
* `torch` library

## Setup

### Clone the repository

### Install dependencies

Navigate into the project directory and install the required libraries using pip:
cd text-summarization-autogen pip install -r requirements.txt

### Alternatively, install each library individually:
pip install transformers autogen torch

## Usage

To run the text summarization script, follow these steps:

### Navigate to the script directory
cd text-summarization-autogen

### Run the script

Execute the script using Python:
python text_summarization.py


## Classes and Methods

### TextSummarizer

A class to handle text summarization using a specified model.

Methods:

* `__init__(self, model_name)`: Initializes the summarizer with a specified model.
* `summarize(self, text)`: Summarizes the input text.

### SummarizationAgent

A class that inherits from `autogen.Agent` to act as an agent for summarization.

Methods:

* `__init__(self, model_name)`: Initializes the agent with a specified summarizer model.
* `act(self, text)`: Uses the summarizer to generate a summary.

## Logic

The script initializes two summarization agents with different models (facebook/bart-large-cnn and t5-base). Each agent is responsible for summarizing the input text. The script then runs both agents on the input text and prints their summaries.

## Expected Outcome

The output should display summaries generated by both the BART and T5 models. For example:
Summary by agent 1: [BART model summary] Summary by agent 2: [T5 model summary]

## Example Output
Summary by agent 1: The quick brown fox jumps over the lazy dog. Summary by agent 2: A fast brown fox leaps over a lazy dog.

This demonstrates how multiple AI agents can be used to generate diverse summaries of the same text input, showcasing the capabilities of local models and the AutoGen library.
