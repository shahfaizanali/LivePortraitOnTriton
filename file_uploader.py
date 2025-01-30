# file_uploader.py

import aiohttp
import os

async def upload_file(
    file_path: str,
    auth_token: str,
    thread_id: str,
) -> dict:
    upload_endpoint = "https://ps-api-dev-ce1b0.ravai.hypelaunch.io/api/threads/uploadFilesForThread"

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field('threadId', thread_id)
        
        file_handle = open(file_path, 'rb')
        form.add_field(
            'file',
            file_handle,
            filename=os.path.basename(file_path),
            content_type='application/octet-stream'
        )

        async with session.post(upload_endpoint, data=form, headers={"x-auth-token": auth_token}) as response:
            file_handle.close()
            response.raise_for_status()
            return await response.json()
