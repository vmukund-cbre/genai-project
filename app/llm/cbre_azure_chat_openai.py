"""
CBRE Azure OpenAI Chat Model with WSO2 Authentication

This module provides a custom Azure OpenAI Chat model that inherits from 
LangChain's AzureChatOpenAI and integrates with CBRE's WSO2 authentication system.

Author: CBRE Digital Technology
Created: 2025
"""

import os
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from .cbre_azureopenai_utils import (
    get_access_token, 
    force_token_refresh, 
    get_token_status
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CBREAzureChatOpenAI(AzureChatOpenAI):
    """
    Custom Azure OpenAI Chat model that uses WSO2 authentication.
    
    This class inherits from LangChain's AzureChatOpenAI and overrides the authentication
    mechanism to use CBRE's WSO2 token-based authentication system instead of standard
    Azure AD authentication.
    
    Features:
    - Automatic WSO2 token management
    - Environment variable configuration
    - Token refresh capabilities
    - Full LangChain compatibility
    - Comprehensive error handling
    
    Example:
        # Basic usage with default configuration
        chat_model = CBREAzureChatOpenAI()
        response = chat_model.invoke("Hello, how can you help me?")
        
        # Custom configuration
        chat_model = CBREAzureChatOpenAI.create_with_custom_config(
            temperature=0.7,
            max_tokens=1500
        )
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the CBRE Azure Chat OpenAI client with WSO2 authentication.
        
        This constructor automatically loads configuration from environment variables
        and sets up WSO2 token-based authentication.
        
        Args:
            **kwargs: Additional keyword arguments passed to AzureChatOpenAI.
                     These will override default values from environment variables.
        
        Raises:
            ValueError: If required environment variables are not set
            Exception: If WSO2 authentication fails
        
        Environment Variables Required:
            - AZURE_OPENAI_ENDPOINT: Azure OpenAI service endpoint
            - AZURE_OPENAI_DEPLOYMENT_NAME: Name of the deployed model
            - AZURE_OPENAI_API_VERSION: API version (optional, defaults to 2024-02-15-preview)
            - TEMPERATURE: Model temperature (optional, defaults to 0.0)
            - WSO2_CONSUMER_KEY: WSO2 consumer key
            - WSO2_CONSUMER_SECRET: WSO2 consumer secret
            - WSO2_TOKEN_ENDPOINT: WSO2 token endpoint
        """
        # Load environment variables
        load_dotenv()
        
        # Set default values from environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        temperature = float(os.getenv("TEMPERATURE", "0.0"))
        
        # Validate required environment variables
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set in environment variables")
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME must be set in environment variables")
        
        # Build configuration with defaults and overrides
        config = {
            "azure_endpoint": azure_endpoint,
            "azure_deployment": deployment_name,
            "api_version": api_version,
            "temperature": temperature,
            "azure_ad_token_provider": self._get_azure_ad_token,
            **kwargs  # Allow overriding any default values
        }
        
        logger.info(f"Initializing CBRE Azure Chat OpenAI with deployment: {deployment_name}")
        logger.debug(f"Using endpoint: {azure_endpoint}")
        logger.debug(f"API version: {api_version}")
        
        # Initialize parent class
        super().__init__(**config)
    
    def _get_azure_ad_token(self) -> str:
        """
        Custom token provider that uses WSO2 authentication.
        
        This method is automatically called by the Azure OpenAI client whenever
        it needs an authentication token. It integrates with the WSO2 token
        management system to provide valid access tokens.
        
        Returns:
            str: The access token for Azure OpenAI authentication
            
        Raises:
            Exception: If WSO2 token generation fails
        """
        try:
            token = get_access_token()
            logger.debug("Successfully retrieved WSO2 token for Azure OpenAI")
            return token
        except Exception as e:
            logger.error(f"Failed to get WSO2 token: {e}")
            raise Exception(f"WSO2 Authentication failed: {e}")
    
    def refresh_token(self) -> None:
        """
        Force refresh the WSO2 authentication token.
        
        This method forces a refresh of the cached WSO2 token, useful when
        you encounter authentication errors or want to ensure you have a
        fresh token.
        
        Raises:
            Exception: If token refresh fails
        """
        try:
            force_token_refresh()
            logger.info("Successfully refreshed WSO2 token")
        except Exception as e:
            logger.error(f"Failed to refresh WSO2 token: {e}")
            raise Exception(f"Token refresh failed: {e}")
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """
        Get the current authentication status including token information.
        
        This method provides detailed information about the current state
        of the WSO2 authentication token, including validity and expiration.
        
        Returns:
            Dict[str, Any]: Token status information containing:
                - status: Token status ('valid', 'expired', 'no_token', 'no_expiry')
                - token: Partial token string (first 30 characters + "...")
                - expiry: Token expiration datetime
                - minutes_remaining: Minutes until token expires
        """
        return get_token_status()
    
    def validate_connection(self) -> Dict[str, Any]:
        """
        Validate the connection to Azure OpenAI service.
        
        This method attempts to make a simple request to validate that the
        authentication and connection are working properly.
        
        Returns:
            Dict[str, Any]: Validation result containing success status and details
        """
        try:
            # Make a simple test request
            test_response = self.invoke("Hello")
            
            return {
                "status": "success",
                "message": "Connection validated successfully",
                "token_status": self.get_authentication_status(),
                "test_response_length": len(test_response.content) if hasattr(test_response, 'content') else 0
            }
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return {
                "status": "failed",
                "message": f"Connection validation failed: {e}",
                "token_status": self.get_authentication_status(),
                "error": str(e)
            }
    
    @classmethod
    def create_default(cls, **override_kwargs) -> 'CBREAzureChatOpenAI':
        """
        Factory method to create a CBRE Azure Chat OpenAI instance with default settings.
        
        This is a convenience method that creates an instance using all default
        values from environment variables, with optional overrides.
        
        Args:
            **override_kwargs: Additional keyword arguments to override defaults
            
        Returns:
            CBREAzureChatOpenAI: Configured instance with default settings
            
        Example:
            # Default configuration
            chat_model = CBREAzureChatOpenAI.create_default()
            
            # Override specific settings
            chat_model = CBREAzureChatOpenAI.create_default(temperature=0.5)
        """
        return cls(**override_kwargs)
    
    @classmethod
    def create_with_custom_config(cls, 
                                 temperature: Optional[float] = None,
                                 max_tokens: Optional[int] = None,
                                 top_p: Optional[float] = None,
                                 frequency_penalty: Optional[float] = None,
                                 presence_penalty: Optional[float] = None,
                                 **kwargs) -> 'CBREAzureChatOpenAI':
        """
        Factory method to create a CBRE Azure Chat OpenAI instance with custom configuration.
        
        This method provides a convenient way to create instances with common
        parameter configurations while maintaining type safety and validation.
        
        Args:
            temperature: Controls randomness (0.0 to 2.0). Lower = more deterministic
            max_tokens: Maximum tokens in response (1 to model limit)
            top_p: Controls nucleus sampling (0.0 to 1.0)
            frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0)
            presence_penalty: Penalty for present tokens (-2.0 to 2.0)
            **kwargs: Additional keyword arguments passed to the constructor
            
        Returns:
            CBREAzureChatOpenAI: Configured instance with custom settings
            
        Example:
            # Creative configuration
            creative_model = CBREAzureChatOpenAI.create_with_custom_config(
                temperature=0.8,
                max_tokens=1500,
                top_p=0.9
            )
            
            # Precise configuration  
            precise_model = CBREAzureChatOpenAI.create_with_custom_config(
                temperature=0.0,
                max_tokens=500,
                frequency_penalty=0.1
            )
        """
        config = {}
        
        # Add parameters if provided
        if temperature is not None:
            config['temperature'] = temperature
        if max_tokens is not None:
            config['max_tokens'] = max_tokens
        if top_p is not None:
            config['top_p'] = top_p
        if frequency_penalty is not None:
            config['frequency_penalty'] = frequency_penalty
        if presence_penalty is not None:
            config['presence_penalty'] = presence_penalty
        
        # Merge with additional kwargs
        config.update(kwargs)
        
        return cls(**config)
    
    @classmethod
    def create_for_use_case(cls, use_case: str, **kwargs) -> 'CBREAzureChatOpenAI':
        """
        Factory method to create instances optimized for specific use cases.
        
        This method provides pre-configured instances optimized for common
        use cases in enterprise applications.
        
        Args:
            use_case: The use case to optimize for. Options:
                - 'analysis': For data analysis and structured responses
                - 'creative': For creative writing and brainstorming
                - 'qa': For question answering and information retrieval
                - 'summarization': For text summarization tasks
                - 'classification': For text classification and categorization
            **kwargs: Additional overrides
            
        Returns:
            CBREAzureChatOpenAI: Instance optimized for the specified use case
            
        Raises:
            ValueError: If use_case is not recognized
            
        Example:
            # For Q&A tasks
            qa_model = CBREAzureChatOpenAI.create_for_use_case('qa')
            
            # For creative tasks
            creative_model = CBREAzureChatOpenAI.create_for_use_case('creative')
        """
        use_case_configs = {
            'analysis': {
                'temperature': 0.1,
                'max_tokens': 2000,
                'top_p': 0.95,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            },
            'creative': {
                'temperature': 0.8,
                'max_tokens': 1500,
                'top_p': 0.9,
                'frequency_penalty': 0.2,
                'presence_penalty': 0.1
            },
            'qa': {
                'temperature': 0.3,
                'max_tokens': 1000,
                'top_p': 0.95,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            },
            'summarization': {
                'temperature': 0.2,
                'max_tokens': 800,
                'top_p': 0.9,
                'frequency_penalty': 0.1,
                'presence_penalty': 0.0
            },
            'classification': {
                'temperature': 0.0,
                'max_tokens': 500,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        }
        
        if use_case not in use_case_configs:
            available_cases = ', '.join(use_case_configs.keys())
            raise ValueError(f"Unknown use_case '{use_case}'. Available options: {available_cases}")
        
        config = use_case_configs[use_case].copy()
        config.update(kwargs)  # Allow overriding predefined values
        
        logger.info(f"Creating CBRE Azure Chat OpenAI instance optimized for '{use_case}' use case")
        return cls(**config)


def create_cbre_chat_model(**kwargs) -> CBREAzureChatOpenAI:
    """
    Convenience function to create a CBRE Azure Chat OpenAI instance.
    
    This is a simple factory function that creates a CBREAzureChatOpenAI
    instance with default settings and optional overrides.
    
    Args:
        **kwargs: Keyword arguments passed to CBREAzureChatOpenAI constructor
        
    Returns:
        CBREAzureChatOpenAI: Configured chat model instance
        
    Example:
        # Default configuration
        chat_model = create_cbre_chat_model()
        
        # With custom temperature
        chat_model = create_cbre_chat_model(temperature=0.7)
    """
    return CBREAzureChatOpenAI.create_default(**kwargs)


def create_multiple_models(configs: Dict[str, Dict[str, Any]]) -> Dict[str, CBREAzureChatOpenAI]:
    """
    Create multiple CBRE chat model instances with different configurations.
    
    This utility function allows you to create several model instances at once
    with different configurations, useful for applications that need multiple
    models for different purposes.
    
    Args:
        configs: Dictionary mapping model names to their configuration dictionaries
        
    Returns:
        Dict[str, CBREAzureChatOpenAI]: Dictionary mapping names to model instances
        
    Example:
        models = create_multiple_models({
            'precise': {'temperature': 0.0, 'max_tokens': 500},
            'creative': {'temperature': 0.8, 'max_tokens': 1500},
            'balanced': {'temperature': 0.5, 'max_tokens': 1000}
        })
        
        response = models['precise'].invoke("What is 2+2?")
    """
    models = {}
    for name, config in configs.items():
        try:
            models[name] = CBREAzureChatOpenAI(**config)
            logger.info(f"Successfully created model '{name}' with config: {config}")
        except Exception as e:
            logger.error(f"Failed to create model '{name}': {e}")
            raise Exception(f"Failed to create model '{name}': {e}")
    
    return models
