# Azure Conversational Language Understanding (CLU)

## Overview

Conversational Language Understanding (CLU) is a cloud-based Azure AI service that applies machine learning intelligence to enable you to build natural language understanding capabilities into your applications. CLU enables your application to understand what a person wants in their own words and contextually determines the intent (what they want to do) and extracts entities (key pieces of information).

This directory contains sample code demonstrating how to use Azure's CLU service to build intelligent applications that can understand user queries expressed in natural language.

## What is Natural Language Understanding?

Natural Language Understanding (NLU) is a branch of artificial intelligence that helps computers understand, interpret, and respond to human language in a way that is both meaningful and useful. NLU goes beyond simple keyword matching to understand the semantics and context of language.

Key components of NLU include:

- **Intent Recognition**: Determining what action the user wants to take
- **Entity Extraction**: Identifying and extracting key pieces of information from user input
- **Context Management**: Maintaining context across conversations
- **Language Generation**: Creating natural-sounding responses

## CLU vs. LUIS (Language Understanding)

CLU is the evolution of LUIS (Language Understanding Intelligent Service). Key improvements include:

- Better accuracy for intent classification and entity extraction
- Enhanced multilingual capabilities
- More contextual understanding
- Improved development experience
- Integration with other Azure Language services

## CLU and Azure AI-102 Certification

The Azure AI-102 certification (Designing and Implementing a Microsoft Azure AI Solution) covers CLU as part of its language service components. Key areas to understand for the exam include:

- Creating and configuring CLU resources
- Developing language understanding models
- Training and evaluating CLU models
- Publishing and consuming CLU models
- Implementing best practices for language understanding
- Integration with Azure Bot Service and other Azure services

## Key Concepts

### Intents

Intents represent the purpose or goal expressed in a user's input. For example, in a restaurant booking system, intents might include:

- BookTable
- CheckAvailability
- GetRestaurantInfo
- RequestSpecialAccommodation
- CancelReservation
- ModifyReservation

### Entities

Entities are important pieces of information in the user's input that are needed to fulfill their request. Examples include:

- NumberOfPeople
- ReservationTime
- CustomerName
- PhoneNumber
- SpecialRequest
- ReservationID

### Utterances

Utterances are example phrases that users might say to express an intent. These are used to train the CLU model to recognize various ways a user might express the same intent.

## Model Development Workflow

1. **Create a CLU Project**:
   - Define language, prediction resource
   - Create project description

2. **Define Schema**:
   - Define intents
   - Define entities

3. **Label Data**:
   - Add example utterances for each intent
   - Label entities in the utterances

4. **Train the Model**:
   - Train model with labeled utterances

5. **Evaluate the Model**:
   - Test with new utterances
   - Review and refine

6. **Deploy the Model**:
   - Deploy to a prediction endpoint
   - Integrate with applications

## Implementation in this Project

This project includes:

1. `nlu_test.py`: A command-line script to test CLU functionality
2. `nlu_app.py`: A Streamlit web application demonstrating CLU capabilities with visualizations

### Prerequisites

- Azure subscription
- Azure Language service resource with CLU capabilities
- Environment variables set up in `.env` file:
  - CLU_PREDICTION_ENDPOINT
  - LANGUAGE_SERVICE_KEY
  - CLU_REQUEST_ID
  - CLU_API_KEY
  - CLU_DEPLOYMENT_NAME
  - CLU_PROJECT_NAME

### Usage

1. Test the CLU model from the command line:
   ```
   python nlu_test.py
   ```

2. Launch the interactive web demo:
   ```
   streamlit run streamlit_app.py
   ```
   Then select "Conversational Language Understanding" from the sidebar.

## Best Practices for CLU Development

1. **Design Intents Carefully**:
   - Keep intents distinct and focused
   - Avoid overlap between intents
   - Use a "None" intent for out-of-scope queries

2. **Provide Diverse Utterances**:
   - Include varied vocabulary and phrasing
   - Use different sentence structures
   - Include common spelling errors and colloquialisms

3. **Entity Extraction**:
   - Label entities consistently across utterances
   - Consider using prebuilt entities where applicable
   - Use phrase lists for domain-specific terminology

4. **Testing and Improvement**:
   - Test with real-world user data
   - Review low-confidence predictions
   - Continuously add new utterances based on user interactions

5. **Integration**:
   - Implement fallback strategies for unrecognized intents
   - Consider multi-turn conversations
   - Combine with other AI services for richer experiences

## Resources for Learning

- [Official Azure Documentation on Conversational Language Understanding](https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/conversational-language-understanding/overview)
- [Azure AI-102 Exam Guide](https://learn.microsoft.com/en-us/certifications/azure-ai-engineer/)
- [Microsoft Learn Modules on CLU](https://learn.microsoft.com/en-us/training/modules/create-language-understanding-solution/)
- [Language Studio](https://language.cognitive.azure.com/) - Azure's web interface for creating and managing language models

## Sample Applications

CLU can be used to build various intelligent applications, including:

- Virtual assistants and chatbots
- Customer service automation
- Voice-controlled systems
- Intelligent search applications
- Process automation systems
- Command and control interfaces

## Integration with Other Azure Services

CLU works well with other Azure services:

- **Azure Bot Framework**: Build conversational interfaces
- **Azure Functions**: Create serverless applications
- **Logic Apps**: Automate workflows
- **Power Platform**: Create low-code/no-code solutions
- **Cognitive Services**: Combine with other AI capabilities
- **Azure Speech Service**: Create voice-enabled interfaces

## Code Examples

Here's a basic example of how to call the CLU service:

```python
def analyze_text(query_text, language="en"):
    """Call CLU to analyze text."""
    headers = {
        'Ocp-Apim-Subscription-Key': language_service_key,
        'Apim-Request-Id': request_id,
        'Content-Type': 'application/json'
    }
    
    data = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "id": "1",
                "text": query_text,
                "language": language,
                "participantId": "1"
            }
        },
        "parameters": {
            "projectName": project_name,
            "verbose": True,
            "deploymentName": deployment_name,
            "stringIndexType": "TextElement_V8"
        }
    }
    
    response = requests.post(prediction_endpoint, headers=headers, data=json.dumps(data))
    return response.json()
```

## Contributing

Feel free to extend this project with:

- Additional intents and entities for different scenarios
- Enhanced visualizations
- Integration with other Azure services
- Multi-turn conversation support
- Multilingual capabilities

## Conclusion

CLU is a powerful service for building natural language understanding into your applications. By understanding intents and extracting entities, it enables a more natural human-computer interaction and opens up possibilities for automation and intelligence in various domains.