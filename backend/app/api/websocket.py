"""
WebSocket handler for real-time training updates
"""
import logging
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import asyncio
from ..models.schemas import TrainingProgress
from .routes import train

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for training updates"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self.active_connections))

    async def broadcast(self, message: dict):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug("Error broadcasting to client: %s", e)
                disconnected.add(connection)
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for training progress updates.
    Path: /ws/training

    Sends the current training status every 1 second during active training,
    or every 2 seconds when idle. Uses object identity (`id()`) to detect
    status changes efficiently — the training thread replaces the global
    TrainingProgress object on each update.
    """
    await manager.connect(websocket)

    try:
        # Send initial status immediately
        current = train.current_training_status
        await websocket.send_json({
            "type": "training_update",
            "data": current.dict(),
        })
        last_obj_id = id(current)

        while True:
            try:
                # Check for client messages (ping/pong) without blocking
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.1,
                    )
                    if message == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass

                # Read the latest status
                current = train.current_training_status
                current_obj_id = id(current)
                is_active = current.status.value in ("training", "processing")

                # Send update if the status object changed (new assignment by training thread)
                if current_obj_id != last_obj_id:
                    await websocket.send_json({
                        "type": "training_update",
                        "data": current.dict(),
                    })
                    last_obj_id = current_obj_id
                elif is_active:
                    # During active training, always send periodic updates
                    # so the frontend stays in sync even if the object didn't change
                    await websocket.send_json({
                        "type": "training_update",
                        "data": current.dict(),
                    })

                # Poll faster during active training
                await asyncio.sleep(1.0 if is_active else 2.0)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning("Error in websocket loop: %s", e)
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        manager.disconnect(websocket)
