import jwt
from aiohttp import web
from db import get_user_by_id
import os

JWT_SECRET = os.getenv("JWT_SECRET", "secretprovid")

@web.middleware
async def is_authenticated_middleware(request, handler):
    if request.method == "OPTIONS":
        return await handler(request)
    token = request.cookies.get("token")
    if not token:
        return web.json_response({"error": "Access denied. No token provided."}, status=401)
    
    try:
        decoded_data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = decoded_data.get("id")
        
        # user = await get_user_by_id(user_id)
        # if not user:
        #     response = web.json_response({"error": "User not found"}, status=401)
        #     response.del_cookie("token")
        #     return response

        # request["user"] = user
        request["user_id"] = user_id
        return await handler(request)
    except jwt.ExpiredSignatureError:
        response = web.json_response({"error": "Token has expired"}, status=401)
        response.del_cookie("token")
        return response
    except jwt.InvalidTokenError:
        response = web.json_response({"error": "Invalid token"}, status=401)
        response.del_cookie("token")
        return response
