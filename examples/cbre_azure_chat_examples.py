"""
Example usage of CBRE Azure Chat OpenAI with WSO2 Authentication

This file demonstrates various ways to use the CBREAzureChatOpenAI class
for different use cases in enterprise applications.

Usage:
    python examples/cbre_azure_chat_examples.py
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

from llm.cbre_azure_chat_openai import (
    CBREAzureChatOpenAI,
    create_cbre_chat_model,
    create_multiple_models
)


def example_basic_usage():
    """Demonstrate basic usage of CBREAzureChatOpenAI."""
    print("🔸 Example 1: Basic Usage")
    print("-" * 50)
    
    try:
        # Create chat model with default settings
        chat_model = CBREAzureChatOpenAI()
        
        # Check authentication status
        auth_status = chat_model.get_authentication_status()
        print(f"Authentication Status: {auth_status['status']}")
        print(f"Token expires in: {auth_status.get('minutes_remaining', 'N/A')} minutes")
        
        # Make a simple request
        response = chat_model.invoke("Hello! Can you introduce yourself?")
        print(f"Response: {response.content[:200]}...")
        
        print("✅ Basic usage successful!")
        
    except Exception as e:
        print(f"❌ Error in basic usage: {e}")


def example_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n🔸 Example 2: Custom Configuration")
    print("-" * 50)
    
    try:
        # Create model with custom settings
        custom_model = CBREAzureChatOpenAI.create_with_custom_config(
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )
        
        response = custom_model.invoke("Write a creative short story about AI in 2 sentences.")
        print(f"Creative Response: {response.content}")
        
        print("✅ Custom configuration successful!")
        
    except Exception as e:
        print(f"❌ Error in custom configuration: {e}")


def example_use_case_optimization():
    """Demonstrate use case specific optimizations."""
    print("\n🔸 Example 3: Use Case Optimization")
    print("-" * 50)
    
    try:
        # Create models optimized for different use cases
        qa_model = CBREAzureChatOpenAI.create_for_use_case('qa')
        analysis_model = CBREAzureChatOpenAI.create_for_use_case('analysis')
        
        # Q&A example
        qa_response = qa_model.invoke("What is the capital of France?")
        print(f"Q&A Response: {qa_response.content}")
        
        # Analysis example
        analysis_response = analysis_model.invoke(
            "Analyze the following data trends: Sales increased 15% in Q1, "
            "decreased 5% in Q2, and increased 20% in Q3."
        )
        print(f"Analysis Response: {analysis_response.content[:200]}...")
        
        print("✅ Use case optimization successful!")
        
    except Exception as e:
        print(f"❌ Error in use case optimization: {e}")


def example_multiple_models():
    """Demonstrate creating multiple models with different configurations."""
    print("\n🔸 Example 4: Multiple Models")
    print("-" * 50)
    
    try:
        # Create multiple models at once
        models = create_multiple_models({
            'precise': {'temperature': 0.0, 'max_tokens': 300},
            'creative': {'temperature': 0.8, 'max_tokens': 500},
            'balanced': {'temperature': 0.5, 'max_tokens': 400}
        })
        
        question = "What are the benefits of renewable energy?"
        
        for model_name, model in models.items():
            response = model.invoke(question)
            print(f"{model_name.title()} Model: {response.content[:150]}...")
        
        print("✅ Multiple models successful!")
        
    except Exception as e:
        print(f"❌ Error with multiple models: {e}")


def example_token_management():
    """Demonstrate token management features."""
    print("\n🔸 Example 5: Token Management")
    print("-" * 50)
    
    try:
        chat_model = CBREAzureChatOpenAI()
        
        # Check detailed authentication status
        auth_status = chat_model.get_authentication_status()
        print("Authentication Status Details:")
        for key, value in auth_status.items():
            print(f"  {key}: {value}")
        
        # Validate connection
        validation = chat_model.validate_connection()
        print(f"\nConnection Validation: {validation['status']}")
        print(f"Message: {validation['message']}")
        
        # Demonstrate token refresh (uncomment if needed)
        # print("\nRefreshing token...")
        # chat_model.refresh_token()
        # print("Token refreshed successfully!")
        
        print("✅ Token management successful!")
        
    except Exception as e:
        print(f"❌ Error in token management: {e}")


def example_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n🔸 Example 6: Error Handling")
    print("-" * 50)
    
    try:
        # Create model
        chat_model = CBREAzureChatOpenAI()
        
        # Simulate various scenarios
        print("Testing normal operation...")
        response = chat_model.invoke("Hello")
        print(f"Normal response received: {len(response.content)} characters")
        
        # Test with invalid use case (should raise ValueError)
        try:
            invalid_model = CBREAzureChatOpenAI.create_for_use_case('invalid_case')
        except ValueError as e:
            print(f"✅ Correctly caught invalid use case error: {e}")
        
        print("✅ Error handling successful!")
        
    except Exception as e:
        print(f"❌ Error in error handling example: {e}")


def example_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n🔸 Example 7: Convenience Functions")
    print("-" * 50)
    
    try:
        # Using convenience function
        simple_model = create_cbre_chat_model(temperature=0.3)
        
        response = simple_model.invoke("Explain quantum computing in simple terms.")
        print(f"Simple explanation: {response.content[:200]}...")
        
        print("✅ Convenience functions successful!")
        
    except Exception as e:
        print(f"❌ Error with convenience functions: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("CBRE Azure Chat OpenAI Examples")
    print("=" * 60)
    
    # Check if environment variables are set
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_DEPLOYMENT_NAME',
        'WSO2_CONSUMER_KEY',
        'WSO2_CONSUMER_SECRET',
        'WSO2_TOKEN_ENDPOINT'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file and ensure all required variables are set.")
        return
    
    print("✅ All required environment variables are set.")
    print()
    
    # Run examples
    example_basic_usage()
    example_custom_configuration()
    example_use_case_optimization()
    example_multiple_models()
    example_token_management()
    example_error_handling()
    example_convenience_functions()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
