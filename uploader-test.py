# main.py

import asyncio
from file_uploader import upload_file

async def main():
    file_path = "/recordings/677f9cdbe328197a8fdbd1bf/b0ae1f99-d929-4876-8b8a-8574b241305e.mp4"
    thread_id = "677f9d42e328197a8fdbd1eb"

    try:
        response_data = await upload_file(
            file_path=file_path,
            thread_id=thread_id,
            auth_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY3N2Y5Y2RiZTMyODE5N2E4ZmRiZDFiZiIsImlhdCI6MTczODIzNjM1NSwiZXhwIjoxNzM4MzIyNzU1fQ.6SE0H4GeyDhpCw2TsUQ3pRSLMSo0R-q-fUqJtfPl_IQ"
        )
        print("Upload succeeded. Server response:", response_data)
    except Exception as exc:
        print("Upload failed:", exc)

if __name__ == "__main__":
    asyncio.run(main())
