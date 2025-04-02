# Azure AI-102 Exam: Text Analytics Study Guide

## Overview of Text Analytics Service

Text Analytics is part of Azure Cognitive Services and provides natural language processing (NLP) features for text analysis, including:

- Sentiment Analysis
- Key Phrase Extraction
- Named Entity Recognition
- Entity Linking
- Language Detection
- Personally Identifiable Information (PII) Detection
- Text Analytics for Health

## Key Concepts for AI-102 Exam

### 1. Provisioning and Authentication

- **Service Creation Methods**:
  - Azure Portal
  - Azure CLI
  - ARM Templates
  - Terraform

- **Authentication Methods**:
  - API Key (`AzureKeyCredential`)
  - Azure Active Directory (`DefaultAzureCredential`)

- **Regional Availability**:
  - Know that Text Analytics is available in multiple regions but not all regions

- **Example Code for Authentication**:
  ```python
  from azure.core.credentials import AzureKeyCredential
  from azure.ai.textanalytics import TextAnalyticsClient
  
  endpoint = "https://your-resource-name.cognitiveservices.azure.com/"
  key = "your-api-key"
  
  credential = AzureKeyCredential(key)
  client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
  ```

### 2. Input Format and Limitations

- **Document Format**:
  - Documents must include `id` and `text` fields
  - Optionally include `language` field
  
- **Service Limitations** (important for exam):
  - Maximum 5 documents per batch request
  - Maximum 125,000 characters per document
  - Maximum 5,120 characters for PII detection per document
  - Rate limits depend on pricing tier

- **Error Handling**:
  - Checking for `is_error` in response objects
  - Implementing retry policies

### 3. Service Features and Functionality

#### Sentiment Analysis

- **Types of Analysis**:
  - Document-level sentiment
  - Sentence-level sentiment
  - Opinion mining/aspect-based sentiment

- **Sentiment Categories**:
  - Positive
  - Neutral
  - Negative
  - Mixed

- **Confidence Scores**: How to interpret the scores returned (0-1 scale)

- **Exam Tips**:
  - Know how to analyze the sentiment of a list of documents
  - Understand how to process sentence-level sentiment
  - Know how to interpret confidence scores

#### Language Detection

- **Response Format**:
  - ISO 6391 language code
  - Language name
  - Confidence score

- **Supported Languages**: Over 120 languages

- **Exam Tips**:
  - Know that you should use language detection before other operations if language is unknown
  - Understand how to handle multi-language documents
  - Know that you can override detected language with specified language code

#### Named Entity Recognition

- **Entity Categories**:
  - Person
  - Location
  - Organization
  - Quantity
  - DateTime
  - URL
  - Email
  - Phone Number
  - IP Address

- **Exam Tips**:
  - Know the different entity categories
  - Understand how to use entity recognition with different languages
  - Know how to interpret confidence scores for entities

#### Entity Linking

- **Linked Entity Information**:
  - Data source (usually Wikipedia)
  - Entity ID
  - URL
  - Name
  - Matches (where entity appears in text)

- **Exam Tips**:
  - Understand the difference between entity recognition and entity linking
  - Know that entity linking provides disambiguated entities with links to Wikipedia

#### Key Phrase Extraction

- **Function**: Extracts the main points from text

- **Response Format**: List of key phrases

- **Exam Tips**:
  - Know how to use key phrase extraction in different languages
  - Understand how key phrases are determined
  - Know that key phrases are returned as a list without scores

#### PII Detection and Redaction

- **PII Categories**:
  - ABA Routing Number
  - Credit Card Number
  - Email
  - Phone Number
  - Social Security Number
  - etc.

- **Domain-specific PII**:
  - Health-related information
  - Financial information

- **Exam Tips**:
  - Know how to detect PII in text
  - Understand how to redact PII from text
  - Know the difference between PII and PHI (Protected Health Information)

### 4. Advanced Features

#### Text Analytics for Health

- **Function**: Extracts health-related information from unstructured text

- **Features**:
  - Medical entity extraction
  - Relation extraction
  - Entity linking to medical knowledge bases

- **Exam Tips**:
  - Know that this is a specialized feature for healthcare
  - Understand basic capabilities but details are less likely to be tested

#### Custom Named Entity Recognition (Custom NER)

- **Function**: Trains custom models to recognize specific entities

- **Process**:
  - Creating a project in Language Studio
  - Labeling entities in your data
  - Training and testing the model
  - Deploying the model

- **Exam Tips**:
  - Know the basic steps for creating a custom NER model
  - Understand when to use custom NER vs. pre-built entity recognition

### 5. SDK vs. REST API

- **SDK Advantages**:
  - Type safety
  - Built-in authentication
  - Better error handling

- **REST API Advantages**:
  - Can be used from any programming language
  - Direct control over requests

- **Exam Tips**:
  - Know basic REST endpoints for Text Analytics
  - Understand how to structure requests in both SDK and REST
  - Know how to handle batch processing in both approaches

### 6. Integration with Other Azure Services

- **Integration with Azure Logic Apps and Power Automate**:
  - Using Text Analytics connectors

- **Integration with Azure Functions**:
  - Serverless processing of text

- **Integration with Azure Synapse Analytics**:
  - Large-scale text analysis

- **Exam Tips**:
  - Know basic integration patterns
  - Understand how to use Text Analytics in end-to-end solutions

### 7. Monitoring and Logging

- **Metrics Available**:
  - Successful calls
  - Error rates
  - Latency
  - Throttled requests

- **Logging Options**:
  - Azure Monitor
  - Application Insights
  - Diagnostic Logs

- **Exam Tips**:
  - Know how to monitor Text Analytics service
  - Understand how to set up alerts for service issues

### 8. Security and Compliance

- **Data Privacy**:
  - Text data is not stored by the service (by default)
  - Options for custom data storage

- **Compliance**:
  - HIPAA, SOC, ISO compliance
  - Data residency considerations

- **Exam Tips**:
  - Know basic security features
  - Understand compliance considerations

## Sample Questions to Test Your Knowledge

1. What is the maximum number of documents that can be sent in a single batch request to Text Analytics?
   - Answer: 5 documents

2. Which authentication method would you use for a production application that needs to access Text Analytics without storing API keys in code?
   - Answer: Azure Active Directory with Managed Identity

3. What Text Analytics feature would you use to find all mentions of people, organizations, and locations in a text?
   - Answer: Named Entity Recognition

4. What would you use to link recognized entities to a knowledge base with additional information?
   - Answer: Entity Linking

5. If you need to analyze text in multiple languages and don't know which languages are present, what should you do first?
   - Answer: Use the Language Detection feature before other operations

6. How would you handle a scenario where you need to process more than 5 documents?
   - Answer: Split the documents into batches of 5 or fewer and make multiple API calls

7. Which of these is NOT returned by sentiment analysis?
   - Answer: Entity information (sentiment analysis returns sentiment labels and confidence scores)

## Code Examples for Key Operations

### Sentiment Analysis
```python
documents = [
    {"id": "1", "text": "I had a wonderful trip to Seattle last week."},
    {"id": "2", "text": "The product quality was poor and customer service was terrible."}
]

response = client.analyze_sentiment(documents=documents)
for result in response:
    print(f"Document ID: {result.id}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Positive score: {result.confidence_scores.positive}")
    print(f"Neutral score: {result.confidence_scores.neutral}")
    print(f"Negative score: {result.confidence_scores.negative}")
```

### Language Detection
```python
documents = [
    {"id": "1", "text": "Hello world"},
    {"id": "2", "text": "Bonjour tout le monde"}
]

response = client.detect_language(documents=documents)
for result in response:
    print(f"Document ID: {result.id}")
    print(f"Language: {result.primary_language.name}")
    print(f"ISO6391 name: {result.primary_language.iso6391_name}")
    print(f"Confidence score: {result.primary_language.confidence_score}")
```

### Entity Recognition
```python
documents = [
    {"id": "1", "text": "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975."}
]

response = client.recognize_entities(documents=documents)
for result in response:
    print(f"Document ID: {result.id}")
    for entity in result.entities:
        print(f"Entity: {entity.text}")
        print(f"Category: {entity.category}")
        print(f"Subcategory: {entity.subcategory}")
        print(f"Confidence score: {entity.confidence_score}")
```

### Key Phrase Extraction
```python
documents = [
    {"id": "1", "text": "The food was delicious and the service was excellent. The atmosphere was charming."}
]

response = client.extract_key_phrases(documents=documents)
for result in response:
    print(f"Document ID: {result.id}")
    print(f"Key phrases: {result.key_phrases}")
```

### PII Detection
```python
documents = [
    {"id": "1", "text": "Call me at 555-555-5555 or email john.doe@example.com"}
]

response = client.recognize_pii_entities(documents=documents)
for result in response:
    print(f"Document ID: {result.id}")
    for entity in result.entities:
        print(f"Entity: {entity.text}")
        print(f"Category: {entity.category}")
        print(f"Confidence score: {entity.confidence_score}")
```

## Best Practices for the Exam

1. **Batch Processing**: Always remember the 5-document limit per batch and implement proper batching

2. **Error Handling**: Check for errors in each document result (`is_error` property)

3. **Language Handling**: Use language detection or specify language when known

4. **Service Limits**: Be aware of character limits, rate limits, and throughput limits

5. **Cost Management**: Understand the pricing tiers and how charges are calculated

6. **Security**: Know how to properly authenticate and protect credentials

7. **Performance**: Understand how to optimize for better performance (batching, async operations)

## Conclusion

Text Analytics is a powerful service within Azure Cognitive Services that provides sophisticated natural language processing capabilities. For the AI-102 exam, focus on understanding the service capabilities, limitations, authentication methods, and integration patterns. Make sure you can write code to use each of the main features and handle common scenarios like batching and error handling.