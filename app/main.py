import asyncio
import signal
import sys
from typing import Optional

from app.config import settings
from app.models.chat_schemas import RabbitMQMessage
from app.services import IChatService, IRabbitMQService
from app.services.executor.chat_service_impl import ChatServiceImpl
from app.services.executor.rabbitmq_service_impl import RabbitMQServiceImpl
from app.utils.logger import logger


class ChatBotListener:
    """Main chat bot listener that handles RabbitMQ messages"""

    def __init__(
        self,
        rabbitmq_service: IRabbitMQService = RabbitMQServiceImpl(),
        chat_service: IChatService = ChatServiceImpl(),
    ):
        self.rabbitmq_service = rabbitmq_service
        self.chat_service = chat_service
        self.running = False

    async def start(self) -> None:
        """Start the chat bot listener"""
        try:
            logger.info("Starting Chat Bot Listener...")

            # Connect to RabbitMQ
            await self.rabbitmq_service.connect()

            # Start consuming messages
            await self.rabbitmq_service.start_consuming(self.handle_message)

            self.running = True
            logger.info("Chat Bot Listener started successfully")

            # Keep the service running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error starting Chat Bot Listener: {str(e)}")
            raise

    async def stop(self) -> None:
        """Stop the chat bot listener"""
        try:
            logger.info("Stopping Chat Bot Listener...")
            self.running = False

            # Stop consuming messages
            await self.rabbitmq_service.stop_consuming()

            # Disconnect from RabbitMQ
            await self.rabbitmq_service.disconnect()

            logger.info("Chat Bot Listener stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping Chat Bot Listener: {str(e)}")

    async def handle_message(self, message: RabbitMQMessage) -> None:
        """Handle incoming RabbitMQ message"""
        try:
            await self.chat_service.handle_rabbitmq_message(message)
        except Exception as e:
            logger.exception(f"Error handling message {message.message_id}: {str(e)}")


# Global listener instance
listener: Optional[ChatBotListener] = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if listener:
        asyncio.create_task(listener.stop())
    sys.exit(0)


async def main():
    """Main function"""
    global listener

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create and start listener
        listener = ChatBotListener()
        await listener.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        if listener:
            await listener.stop()


if __name__ == "__main__":
    # Set up logging
    logger.info("Starting AI Agents Chat Bot...")
    logger.info(f"Configuration:")
    logger.info(f"  - RabbitMQ: {settings.rabbitmq_host}:{settings.rabbitmq_port}")
    logger.info(f"  - Redis: {settings.redis_host}:{settings.redis_port}")

    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)
