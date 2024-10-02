from aiohttp import web
import asyncio

async def duplicate_data(request):
    data = await request.json()
    duplicated_data = data['data'] + '\n' + data['data']
    return web.json_response({'duplicated_data': duplicated_data})

async def main():
    app = web.Application()
    app.router.add_post('/duplicate', duplicate_data)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8081)
    await site.start()

    print("Program 2 is running on http://localhost:8081")

    await asyncio.sleep(100*3600)  # Keeps the server running

asyncio.run(main())
