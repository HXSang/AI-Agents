import asyncio

from clients.rabbitmq_client import RabbitMQClient
from config import settings
from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from middleware.auth import verify_bearer_token
from routers import (
    calendar_router,
    chat_router,
    connection_router,
    financial_router,
    health_data_router,
    health_router,
    insight_router,
    monthly_insight_router,
    onboarding_router,
)
from services.executor.chat_service_impl import ChatServiceImpl
from services.service_factory import ServiceFactory
from utils.exception_handlers import (
    generic_exception_handler,
    type_error_handler,
    validation_exception_handler,
    value_error_handler,
)
from utils.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Backend service for chatbot communication with AI agents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Health router - NO AUTH (public health check)
app.include_router(health_router.router)

# Protected routers - REQUIRE BEARER TOKEN
app.include_router(
    chat_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    connection_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    onboarding_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    health_data_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    insight_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    monthly_insight_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    financial_router.router,
    dependencies=[Depends(verify_bearer_token)],
)
app.include_router(
    calendar_router.router,
    dependencies=[Depends(verify_bearer_token)],
)


# Register common exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(TypeError, type_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Daily insight (health / productivity / overall) uses AdvancedInsightService
# (Q&A pipeline). Chat-driven override is intentionally disabled so insight
# does not go through RabbitMQ / HybridWorkflow.

# Insight Demo - Public route (no auth required)
@app.get("/insight-demo", include_in_schema=False)
async def insight_demo():
    """Serve the insight demo HTML page"""
    return FileResponse("scratch/insight_demo.html")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting chatbot backend service...")

        # Set external API service FIRST (independent of RabbitMQ)
        # This ensures services always have external_api_service even if RabbitMQ fails
        try:
            external_api_service = ServiceFactory.get_external_api_service()

            # Set external API service for chat service
            chat_service = ServiceFactory.get_chat_service()
            chat_service.set_external_api_service(external_api_service)

            # Set external API service for insight service
            insight_service = ServiceFactory.get_insight_service()
            insight_service.set_external_api_service(external_api_service)

            logger.info("✅ External API service initialized and set for all services")
        except Exception as external_api_error:
            logger.error(
                f"❌ Failed to initialize external API service: {str(external_api_error)}"
            )
            logger.warning("⚠️ Services will run without external API functionality")
            # Continue anyway - services will handle None checks

        # Try to initialize RabbitMQ client
        try:
            rabbitmq_client = RabbitMQClient()
            await rabbitmq_client.connect()

            # Set RabbitMQ client for chat and init_session services
            chat_service = ServiceFactory.get_chat_service()
            chat_service.set_rabbitmq_client(rabbitmq_client)

            init_session_service = ServiceFactory.get_init_session_service()
            init_session_service.set_rabbitmq_client(rabbitmq_client)

            # Response queue consumer is disabled for chat flow.
            # App listener now handles post-workflow processing and writes response to Redis.

            logger.info("🚀 Backend service started successfully with RabbitMQ")

        except Exception as rabbitmq_error:
            logger.warning(f"⚠️ RabbitMQ connection failed: {str(rabbitmq_error)}")
            logger.warning("Running in development mode without RabbitMQ")
            logger.warning("Chat functionality will be limited")

    except Exception as e:
        logger.error(f"❌ Failed to start backend service: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        rabbitmq_client = RabbitMQClient()
        await rabbitmq_client.stop_consuming()
        await rabbitmq_client.disconnect()

        logger.info("🛑 Backend service stopped")

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
    )
