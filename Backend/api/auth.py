from fastapi import APIRouter, HTTPException, Depends, status, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional
import logging

from .config import settings
from .supabase_client import supabase, validate_token

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ... existing code ...

@router.get("/validate-token")
async def validate_auth_token(authorization: str = Header(None)):
    """
    Validate a JWT token by checking with Supabase.
    Returns 200 OK if valid, 401 if invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        logger.error("No or invalid authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid authentication credentials"
        )
    
    token = authorization.replace("Bearer ", "")
    
    try:
        # Validate the token with Supabase
        user_id = await validate_token(token)
        if user_id:
            return {"valid": True, "user_id": user_id}
        else:
            logger.error("Token validation failed: invalid token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token"
            )
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Token validation error: {str(e)}"
        ) 