from io import BufferedReader
from datetime import datetime
import asyncio
import telegram

class TeleBot:
    def __init__(self, token : str, chats = None, addtime=False) -> None:
        self.bot = telegram.Bot(token)
        self.chats = []
        if isinstance(chats, list):
            self.chats = []
        elif isinstance(chats, int):
            self.chats.append(chats)
        self.timestamp = addtime

    def add_chat(self, chatid : int):
        self.chats.append(chatid)

    async def _sendmsg(self, msg : str, chatid : int):
        async with self.bot:
            await self.bot.send_message(text=msg, chat_id=chatid)

    async def _sendfile(self, file : BufferedReader | bytes, chatid : int, caption : str | None):
        async with self.bot:
            await self.bot.send_document(chatid, file, caption)

    def message(self, msg : str):
        if self.timestamp:
            msg = f'{datetime.now()}: ' + msg
        for chat in self.chats:
            asyncio.run(self._sendmsg(msg, chat))

    def sendfile(self, file : str | BufferedReader | bytes, caption : str = None):
        if isinstance(file, str):
            file = open(file, 'rb')
        elif not isinstance(file, bytes | BufferedReader):
            raise ValueError('Incorrect argument type (file)')
        for chat in self.chats:
            asyncio.run(self._sendfile(file, chat, caption))

if __name__ == "__main__":
    bot = TeleBot("5811298447:AAF0--61uBVvKgFvMeYs76fB1QjmhaihU-Y", -822387173)
    bot.message('Hello!')
