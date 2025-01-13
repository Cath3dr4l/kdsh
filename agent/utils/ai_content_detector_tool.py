from langchain.tools import BaseTool
import requests
import asyncio
import aiohttp


class TextDetectionTool(BaseTool):
    name: str = "text_detection"
    description: str = (
        "Detects if the input text is fake or real by analyzing text segments."
    )
    api_url: str = "https://api.zerogpt.com/api/detect/detectText"
    headers: dict = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8,te;q=0.7",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Cookie": "_ga=GA1.1.731630520.1736694648; _ga_0YHYR2F422=GS1.1.1736694648.1.1.1736694976.0.0.0",
        "Host": "api.zerogpt.com",
        "Origin": "https://www.zerogpt.com",
        "Referer": "https://www.zerogpt.com/",
        "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }
    max_chunk_size: int = 6000

    def __init__(self, max_chunk_size=6000):
        super().__init__()
        print(self.api_url)
        # self.api_url = api_url
        # self.headers = headers
        self.max_chunk_size = max_chunk_size

    def split_text(self, text, max_chunk_size):
        """Split the input text into chunks of max_chunk_size."""
        return [
            text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)
        ]

    async def call_api_async(self, session, text_chunk):
        """Send a single text chunk to the API and return the response."""
        print("Calling API")
        payload = {"input_text": text_chunk}
        try:
            async with session.post(self.api_url, json=payload, headers=self.headers) as response:
                response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"API call failed: {e}")
            return {"error": str(e)}  # Return error message in a structured format

    def _run(self, text):
        return asyncio.run(self._run_async(text))
    
    async def _run_async(self, text):
        """Run the text detection tool on the input text asynchronously."""
        # Split the text into chunks
        text_chunks = self.split_text(text, self.max_chunk_size)
        chunks = []
        for chunk in text_chunks:
            if len(chunk) > 2000:
                chunks.append(chunk)
        text_chunks = chunks
        print([len(chunk) for chunk in text_chunks])

        async with aiohttp.ClientSession() as session:
            tasks = [self.call_api_async(session, chunk) for chunk in text_chunks]
            results = await asyncio.gather(*tasks)

        # Aggregate results (e.g., calculate average fake percentage if applicable)
        valid_results = [r["data"]["fakePercentage"] for r in results if "data" in r and "fakePercentage" in r["data"]]
        if valid_results:
            average_fake_percentage = sum(valid_results) / len(valid_results)
            return {
                "average_fake_percentage": average_fake_percentage,
                "individual_scores": valid_results,
            }
        else:
            return {"error": "Failed to process text", "individual_scores": results}
        
