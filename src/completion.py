from enum import Enum
from dataclasses import dataclass
import openai
import json
import tiktoken
from src.moderation import moderate_message
from typing import Optional, List
from src.constants import (
    BOT_INSTRUCTIONS,
    BOT_NAME,
    EXAMPLE_CONVOS,
    MAX_TOTAL_TOKENS
)

import src.constants
import discord
from src.base import Message, Prompt, Conversation
from src.utils import split_into_shorter_messages, close_thread, logger
from src.moderation import (
    send_moderation_flagged_message,
    send_moderation_blocked_message,
)

MY_BOT_NAME = BOT_NAME
MY_BOT_EXAMPLE_CONVOS = EXAMPLE_CONVOS


class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5


@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]


async def generate_completion_response(
    messages: List[Message], user: str
) -> CompletionData:
    try:
        prompt = Prompt(
            header=Message(
                "System", f"Instructions for {MY_BOT_NAME}: {BOT_INSTRUCTIONS}"
            ),
            examples=MY_BOT_EXAMPLE_CONVOS,
            convo=Conversation(messages + [Message(MY_BOT_NAME)]),
        )
        rendered = prompt.render()
        logger.info(f"Prompt: {prompt}")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=rendered,
            temperature=1.0,
            top_p=0.9,
            max_tokens=512,
            stop=["<|endoftext|>"],
        )
        reply = response.choices[0].text.strip()
        if reply:
            flagged_str, blocked_str = moderate_message(
                message=(rendered + reply)[-500:], user=user
            )
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    except openai.error.InvalidRequestError as error:
        if "This model's maximum context length" in error.user_message:
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(error)
            )
        else:
            logger.exception(error)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(error),
            )
    except Exception as error:
        logger.exception(error)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(error)
        )

async def generate_chat_response(
    messages: List[Message], user: str
) -> CompletionData:
    try:

        #reverse messages
        messages.reverse()              
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        max_response_tokens=512
        finalPrompt = []
        for message in messages:
            # Get the values of "role" and "content" from the message dictionary
            if message.user == "ChatBot":
                role = "assistant"
            else:
                role = "user"
            content = message.text          
            # Create a new dictionary with "role" and "content" keys
            newMessage = {"role": role, "content": content}
            #count number of tokens in newMessage
            total_message_num_tokens = len(encoding.encode(json.dumps(finalPrompt)))
            current_message_num_tokens = len(encoding.encode(json.dumps(newMessage)))
            if(total_message_num_tokens + current_message_num_tokens > MAX_TOTAL_TOKENS - max_response_tokens):
                break
            # Append the new dictionary to the finalPrompt list
            finalPrompt.insert(0,newMessage)
            
        
        logger.info(f"Number of tokens: {total_message_num_tokens}	")
        MAX_TOTAL_TOKENS
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=finalPrompt
        )
        logger.info(json.dumps(finalPrompt,indent=4))
    
        reply = response.choices[0].message.content
        if reply:
            flagged_str, blocked_str = moderate_message(
                message=(json.dumps(finalPrompt) + reply)[-500:], user=user
            )
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    except openai.error.InvalidRequestError as error:
        if "This model's maximum context length" in error.user_message:
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(error)
            )
        else:
            logger.exception(error)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(error),
            )
    except Exception as error:
        logger.exception(error)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(error)
        )


async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData
):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text
    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        sent_message = None
        if not reply_text:
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)
        if status is CompletionResult.MODERATION_FLAGGED:
            await send_moderation_flagged_message(
                guild=thread.guild,
                user=user,
                flagged_str=status_text,
                message=reply_text,
                url=sent_message.jump_url if sent_message else "no url",
            )

            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **This conversation has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )
    elif status is CompletionResult.MODERATION_BLOCKED:
        await send_moderation_blocked_message(
            guild=thread.guild,
            user=user,
            blocked_str=status_text,
            message=reply_text,
        )

        await thread.send(
            embed=discord.Embed(
                description=f"❌ **The response has been blocked by moderation.**",
                color=discord.Color.red(),
            )
        )
    elif status is CompletionResult.TOO_LONG:
        await close_thread(thread)
    elif status is CompletionResult.INVALID_REQUEST:
        await thread.send(
            embed=discord.Embed(
                description=f"**Invalid request** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    else:
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )