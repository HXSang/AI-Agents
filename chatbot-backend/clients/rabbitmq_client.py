import asyncio
import json
import ssl
from typing import Callable, Dict, Optional

import aio_pika
from aio_pika import Connection, Exchange, Message, Queue
from aio_pika.abc import AbstractIncomingMessage
from config import settings
from models.models import RabbitMQMessage


class RabbitMQClient:
    """RabbitMQ client for sending and receiving messages"""

    def __init__(self):
        self.connection: Optional[Connection] = None
        self.channel = None
        self.input_exchange: Optional[Exchange] = None
        self.response_exchange: Optional[Exchange] = None
        self.response_queue: Optional[Queue] = None
        self.consuming = False
        self.response_semaphore = asyncio.Semaphore(6)

    async def connect(self) -> None:
        """Connect to RabbitMQ"""
        try:
            # Determine protocol based on port (5671 = SSL, 5672 = non-SSL)
            protocol = "amqps" if settings.rabbitmq_port == 5671 else "amqp"

            # Create connection URL
            connection_url = (
                f"{protocol}://{settings.rabbitmq_username}:{settings.rabbitmq_password}"
                f"@{settings.rabbitmq_host}:{settings.rabbitmq_port}{settings.rabbitmq_vhost}"
            )

            # For Amazon MQ (SSL), create SSL context that doesn't verify certificates
            ssl_context = None
            if protocol == "amqps":
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            self.connection = await aio_pika.connect_robust(
                connection_url, ssl_context=ssl_context
            )
            self.channel = await self.connection.channel()

            # Set QoS
            await self.channel.set_qos(prefetch_count=8)

            # Declare exchanges
            self.input_exchange = await self.channel.declare_exchange(
                "chat_input", aio_pika.ExchangeType.DIRECT, durable=True
            )
            self.response_exchange = await self.channel.declare_exchange(
                "chat_answer", aio_pika.ExchangeType.DIRECT, durable=True
            )

            # Declare response queue for receiving answers
            self.response_queue = await self.channel.declare_queue(
                settings.response_queue, durable=True
            )

            # Bind response queue to exchange with correct routing key
            await self.response_queue.bind(
                self.response_exchange, settings.response_queue
            )

            print("✅ Connected to RabbitMQ successfully")

        except Exception as e:
            protocol = "amqps" if settings.rabbitmq_port == 5671 else "amqp"
            print(f"❌ Failed to connect to RabbitMQ: {str(e)}")
            print(
                f"Connection URL: {protocol}://{settings.rabbitmq_username}:***@{settings.rabbitmq_host}:{settings.rabbitmq_port}{settings.rabbitmq_vhost}"
            )
            print("Please check:")
            print("1. RabbitMQ server is running")
            print("2. Network connectivity to the server")
            print("3. Credentials are correct")
            print("4. Firewall settings")
            print("5. Server is accessible from your network")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ"""
        try:
            self.consuming = False
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            print("✅ Disconnected from RabbitMQ")
        except Exception as e:
            print(f"❌ Error disconnecting from RabbitMQ: {str(e)}")

    async def send_message(self, message: RabbitMQMessage) -> None:
        """Send message to AI agents"""
        try:
            if not self.input_exchange:
                raise Exception("Not connected to RabbitMQ")

            # Create message
            message_data = message.model_dump(mode="json")
            rabbitmq_message = Message(
                body=json.dumps(message_data).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            # Publish message
            await self.input_exchange.publish(
                rabbitmq_message, routing_key=settings.input_queue
            )

            print(f"📤 Sent message {message.message_id} to AI agents")

        except Exception as e:
            print(f"❌ Error sending message: {str(e)}")
            raise

    async def start_consuming_responses(
        self, response_handler: Callable[[Dict], None]
    ) -> None:
        """Start consuming responses from AI agents"""
        try:
            if not self.response_queue:
                raise Exception("Not connected to RabbitMQ")

            async def process_response(message: AbstractIncomingMessage):
                try:
                    response_data = json.loads(message.body.decode())
                    message_id = response_data.get("message_id", "unknown")

                    print(f"📥 Received response: {message_id}")

                    async def handle_and_ack():
                        async with self.response_semaphore:
                            try:
                                await response_handler(response_data)
                                await message.ack()
                            except Exception as e:
                                print(
                                    f"❌ Error handling response {message_id}: {str(e)}"
                                )
                                await message.nack(requeue=False)

                    # 🚀 Không block consumer
                    asyncio.create_task(handle_and_ack())

                except Exception as e:
                    print(f"❌ Error parsing response: {str(e)}")
                    await message.nack(requeue=False)

            # Start consuming
            await self.response_queue.consume(process_response)
            self.consuming = True
            print("✅ Started consuming responses from AI agents")

        except Exception as e:
            print(f"❌ Error starting response consumption: {str(e)}")
            raise

    async def stop_consuming(self) -> None:
        """Stop consuming responses"""
        try:
            self.consuming = False
            if self.response_queue:
                await self.response_queue.cancel()
            print("✅ Stopped consuming responses")
        except Exception as e:
            print(f"❌ Error stopping response consumption: {str(e)}")

    async def health_check(self) -> bool:
        try:
            if not self.connection:
                return False

            if self.connection.is_closed:
                return False

            if not self.channel or self.channel.is_closed:
                return False

            return True

        except Exception as e:
            print(f"❌ RabbitMQ health check failed: {e}")
            return False
