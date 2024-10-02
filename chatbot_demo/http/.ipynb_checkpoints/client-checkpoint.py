import aiohttp
import asyncio
import streamlit as st

async def send_data_to_prog2(data):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8081/duplicate', json={'data': data}) as response:
            return await response.json()

async def main():
    user_input = st.text_input("Enter your string:")
    response = await send_data_to_prog2(user_input)
    st.write("Program 2 responded:", response)

asyncio.run(main())
