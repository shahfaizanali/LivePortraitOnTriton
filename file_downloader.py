import os
from urllib.parse import urlparse
import aiohttp

TARGET_DIRECTORY = os.getenv("TARGET_DIRECTORY", "/images")

async def download_file(url: str):
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.lstrip("/")  

        if not path:
            raise ValueError(f"Unable to determine file path from URL: {url}")

        full_path = os.path.join(TARGET_DIRECTORY, path)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(full_path, "wb") as f:
                        f.write(await response.read())
                    print(f"File downloaded: {full_path}")
                    return full_path
                else:
                    print(f"Failed to download file from {url}, status: {response.status}")
                    return None
    except Exception as e:
        print(f"Error downloading file from {url}: {e}")
        return None
