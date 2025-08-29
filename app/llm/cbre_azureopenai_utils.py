import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables to store token and expiry
_access_token = None
_token_expiry = None

def get_access_token():
    """
    Generate a new access token using WSO2 consumer key and secret.
    This method generates the Bearer token to connect to WSO2. 
    In general, Bearer token is valid for one hour.
    """
    global _access_token, _token_expiry
    
    # Check if we have a valid cached token
    if _access_token and _token_expiry and datetime.now() < _token_expiry:
        return _access_token
    
    # Get credentials from environment variables
    client_id = os.getenv("WSO2_CONSUMER_KEY")
    client_secret = os.getenv("WSO2_CONSUMER_SECRET")
    auth_token_endpoint = os.getenv("WSO2_TOKEN_ENDPOINT")
    
    if not client_id or not client_secret:
        raise ValueError("WSO2_CONSUMER_KEY and WSO2_CONSUMER_SECRET must be set in environment variables")
    
    if not auth_token_endpoint:
        raise ValueError("WSO2_TOKEN_ENDPOINT must be set in environment variables")
    
    try:
        response = requests.post(auth_token_endpoint, data={
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        })
        
        if response.status_code == 200:
            token_data = response.json()
            _access_token = token_data['access_token']
            
            # Set expiry time (default to 1 hour minus configurable buffer for safety)
            expires_in = token_data.get('expires_in', 3600)  # Default to 1 hour
            buffer_seconds = int(os.getenv('TOKEN_REFRESH_BUFFER', 300))  # Default 5 minutes buffer
            _token_expiry = datetime.now() + timedelta(seconds=expires_in - buffer_seconds)
            
            print(f"✅ New access token generated, expires at: {_token_expiry}")
            return _access_token
        else:
            raise Exception(f'Failed to obtain access token: HTTP {response.status_code} - {response.text}')
    
    except Exception as e:
        print(f"❌ Error generating access token: {e}")
        raise e

def get_cached_token():
    """
    Get the currently cached token without forcing a refresh.
    Returns None if no valid token is cached.
    """
    global _access_token, _token_expiry
    
    if _access_token and _token_expiry and datetime.now() < _token_expiry:
        return _access_token
    return None

def force_token_refresh():
    """
    Force a refresh of the access token.
    """
    global _access_token, _token_expiry
    _access_token = None
    _token_expiry = None
    return get_access_token()

def get_token_expiry():
    """
    Get the current token expiry time.
    Returns None if no token is cached.
    """
    global _token_expiry
    return _token_expiry

def get_token_status():
    """
    Get comprehensive token status information.
    """
    global _access_token, _token_expiry
    
    if not _access_token:
        return {"status": "no_token", "token": None, "expiry": None, "minutes_remaining": None}
    
    if not _token_expiry:
        return {"status": "no_expiry", "token": _access_token[:30] + "...", "expiry": None, "minutes_remaining": None}
    
    now = datetime.now()
    if now >= _token_expiry:
        return {"status": "expired", "token": _access_token[:30] + "...", "expiry": _token_expiry, "minutes_remaining": 0}
    
    minutes_remaining = (_token_expiry - now).total_seconds() / 60
    return {
        "status": "valid", 
        "token": _access_token[:30] + "...", 
        "expiry": _token_expiry, 
        "minutes_remaining": round(minutes_remaining, 1)
    }

