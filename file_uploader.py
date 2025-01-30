# file_uploader.py

import aiohttp
import os
from typing import Optional, Dict

async def upload_file(
    file_path: str,
    auth_token: str,
    thread_id: str,
) -> dict:
    upload_endpoint = "https://ps-dev-ce1b0.ravai.hypelaunch.io/api/threads/uploadFilesForThread"

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field('threadId', thread_id)
        
        with open(file_path, 'rb') as file_object:
            form.add_field(
                'file',
                file_object,
                filename=os.path.basename(file_path),
                content_type='application/octet-stream'
            )

        async with session.post(upload_endpoint, data=form, headers={"x-auth-token": auth_token}) as response:
            response.raise_for_status()
            return await response.json()
