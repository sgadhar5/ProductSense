import asyncio
from twikit import Client

async def main():
    client = Client("en-US")
    client.load_cookies("cookies_twikit.json")
    me = await client.me()
    print(me)

asyncio.run(main())
