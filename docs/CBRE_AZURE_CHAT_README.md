# CBRE Azure OpenAI Chat Model with WSO2 Authentication

This module provides a custom Azure OpenAI Chat model that integrates with CBRE's WSO2 authentication system while maintaining full compatibility with LangChain's `AzureChatOpenAI`.

## 📁 File Structure

```
app/llm/
├── cbre_azureopenai_utils.py     # WSO2 token management utilities
├── cbre_azure_chat_openai.py     # Main CBREAzureChatOpenAI class
└── ...

examples/
└── cbre_azure_chat_examples.py  # Usage examples and demonstrations
```

## 🚀 Quick Start

### Basic Usage

```python
from app.llm.cbre_azure_chat_openai import CBREAzureChatOpenAI

# Create chat model with default settings
chat_model = CBREAzureChatOpenAI()

# Use like any LangChain chat model
response = chat_model.invoke("Hello, how can you help me?")
print(response.content)
```

### Custom Configuration

```python
# Create model with specific parameters
custom_model = CBREAzureChatOpenAI.create_with_custom_config(
    temperature=0.7,
    max_tokens=1500,
    top_p=0.9
)

response = custom_model.invoke("Write a creative story about AI.")
```

### Use Case Optimization

```python
# Create models optimized for specific tasks
qa_model = CBREAzureChatOpenAI.create_for_use_case('qa')
creative_model = CBREAzureChatOpenAI.create_for_use_case('creative')
analysis_model = CBREAzureChatOpenAI.create_for_use_case('analysis')

# Use appropriate model for each task
answer = qa_model.invoke("What is machine learning?")
story = creative_model.invoke("Write a short story about robots.")
insights = analysis_model.invoke("Analyze this data: [your data]")
```

## 🔧 Configuration

### Required Environment Variables

Create a `.env` file in your app directory with the following variables:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT='https://your-endpoint.com'
AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4o'
AZURE_OPENAI_API_VERSION='2024-02-15-preview'
TEMPERATURE='0.0'

# WSO2 Authentication
WSO2_CONSUMER_KEY='your-consumer-key'
WSO2_CONSUMER_SECRET='your-consumer-secret'
WSO2_TOKEN_ENDPOINT='https://your-wso2-endpoint.com/token'
```

### Optional Configuration

```bash
# Token refresh buffer (seconds before expiry to refresh)
TOKEN_REFRESH_BUFFER='300'

# Text embedding model (if using embeddings)
TEXT_EMBEDDING_MODEL='text-embedding-3-large'
```

## 🎯 Use Cases and Optimization

The library provides pre-configured models for common enterprise use cases:

| Use Case | Temperature | Max Tokens | Best For |
|----------|-------------|------------|----------|
| `analysis` | 0.1 | 2000 | Data analysis, structured responses |
| `qa` | 0.3 | 1000 | Question answering, information retrieval |
| `summarization` | 0.2 | 800 | Text summarization, content condensing |
| `classification` | 0.0 | 500 | Text classification, categorization |
| `creative` | 0.8 | 1500 | Creative writing, brainstorming |

## 🔐 Authentication Features

### Automatic Token Management
- Automatic WSO2 token generation and caching
- Token expiry monitoring and refresh
- Seamless integration with Azure OpenAI

### Token Status Monitoring
```python
# Check authentication status
status = chat_model.get_authentication_status()
print(f"Status: {status['status']}")
print(f"Minutes remaining: {status['minutes_remaining']}")

# Validate connection
validation = chat_model.validate_connection()
print(f"Connection: {validation['status']}")
```

### Manual Token Management
```python
# Force token refresh
chat_model.refresh_token()

# Get detailed token information
from app.llm.cbre_azureopenai_utils import get_token_status
status = get_token_status()
```

## 🏭 Factory Methods

### Default Creation
```python
# Using class method
model = CBREAzureChatOpenAI.create_default()

# Using convenience function
model = create_cbre_chat_model()
```

### Custom Configuration
```python
model = CBREAzureChatOpenAI.create_with_custom_config(
    temperature=0.5,
    max_tokens=1000,
    frequency_penalty=0.1,
    presence_penalty=0.1
)
```

### Multiple Models
```python
from app.llm.cbre_azure_chat_openai import create_multiple_models

models = create_multiple_models({
    'precise': {'temperature': 0.0, 'max_tokens': 500},
    'creative': {'temperature': 0.8, 'max_tokens': 1500},
    'balanced': {'temperature': 0.5, 'max_tokens': 1000}
})

# Use specific model
response = models['precise'].invoke("Calculate 2+2")
```

## 🔍 Advanced Features

### Connection Validation
```python
# Test the connection and authentication
validation_result = chat_model.validate_connection()

if validation_result['status'] == 'success':
    print("✅ Connection is working properly")
else:
    print(f"❌ Connection failed: {validation_result['message']}")
```

### Error Handling
```python
try:
    chat_model = CBREAzureChatOpenAI()
    response = chat_model.invoke("Your question here")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Authentication or network error: {e}")
```

## 📊 Model Parameters

### Temperature Settings
- `0.0` - Completely deterministic responses
- `0.1-0.3` - Mostly consistent with slight variation
- `0.4-0.6` - Balanced creativity and consistency
- `0.7-0.9` - Creative and varied responses
- `1.0+` - Highly creative but potentially unpredictable

### Token Limits
- `max_tokens`: Maximum tokens in response (model dependent)
- Typical ranges: 500-4000 tokens
- 1 token ≈ 0.75 words in English

### Advanced Parameters
- `top_p`: Nucleus sampling (0.0 to 1.0)
- `frequency_penalty`: Reduces repetition (-2.0 to 2.0)
- `presence_penalty`: Encourages topic diversity (-2.0 to 2.0)

## 🧪 Testing and Examples

Run the comprehensive examples:

```bash
cd /path/to/your/project
python examples/cbre_azure_chat_examples.py
```

The examples demonstrate:
- Basic usage patterns
- Custom configurations
- Use case optimizations
- Token management
- Error handling
- Multiple model creation

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via Poetry
   ```bash
   poetry install
   ```

2. **Environment Variables**: Check that all required variables are set
   ```python
   import os
   required_vars = ['AZURE_OPENAI_ENDPOINT', 'WSO2_CONSUMER_KEY', ...]
   missing = [var for var in required_vars if not os.getenv(var)]
   print(f"Missing: {missing}")
   ```

3. **Authentication Failures**: Check WSO2 credentials and endpoint
   ```python
   from app.llm.cbre_azureopenai_utils import get_access_token
   token = get_access_token()  # This will show detailed error messages
   ```

4. **Network Issues**: Verify connectivity to Azure OpenAI and WSO2 endpoints

### Debug Mode
Enable debug logging for detailed information:

```python
import logging
logging.getLogger('app.llm.cbre_azure_chat_openai').setLevel(logging.DEBUG)
```

## 📝 Best Practices

1. **Model Selection**: Choose the right model configuration for your use case
2. **Token Management**: Monitor token expiry and handle refresh gracefully
3. **Error Handling**: Always wrap API calls in try-catch blocks
4. **Configuration**: Use environment variables for sensitive information
5. **Testing**: Use the validation methods before production deployment

## 🤝 Contributing

When making changes to this module:

1. Update documentation for any new features
2. Add examples for new functionality
3. Ensure error handling is comprehensive
4. Test with various configuration scenarios
5. Follow the established naming conventions

## 📄 License

This code is proprietary to CBRE and should only be used within CBRE systems and applications.
