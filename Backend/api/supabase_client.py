import httpx
import logging
from .config import settings

# Set up logger
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = settings.SUPABASE_URL
supabase_key = settings.SUPABASE_KEY

async def validate_token(token):
    """
    Validate the provided token by making a request to Supabase auth API
    Returns user_id if valid, None if invalid
    """
    try:
        # Get user information using the token
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {token}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{supabase_url}/auth/v1/user",
                headers=headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"Successfully validated token for user: {user_data.get('id')}")
                return user_data.get('id')
            else:
                logger.error(f"Error validating token: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        return None 