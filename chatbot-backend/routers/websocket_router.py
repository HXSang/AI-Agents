from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    try:
        websocket_service = ServiceFactory.get_websocket_service()
        await websocket_service.connect(websocket, session_id)

        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()

        except WebSocketDisconnect:
            websocket_service.disconnect(session_id)
        except Exception as e:
            logger.error(f"WebSocket error for session {session_id}: {str(e)}")
            websocket_service.disconnect(session_id)

    except Exception as e:
        logger.error(f"Error in websocket_endpoint: {str(e)}")
        raise
