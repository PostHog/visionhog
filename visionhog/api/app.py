from fastapi import FastAPI, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import pathlib
from ..rtmp_server.server import RTMPServer

# Set up templates directory
current_dir = pathlib.Path(__file__).parent
templates_dir = current_dir.parent.parent / "static" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

def create_app(rtmp_server: RTMPServer) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="VisionHog",
        description="RTMP Server with web interface for video streaming",
        version="0.1.0"
    )
    
    # Mount static files
    static_dir = current_dir.parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Mount HLS directory
    hls_dir = pathlib.Path("./hls")
    os.makedirs(hls_dir, exist_ok=True)
    app.mount("/hls", StaticFiles(directory=str(hls_dir)), name="hls")
    
    # Routes
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Main page showing active streams"""
        streams = rtmp_server.get_active_streams()
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "streams": streams}
        )
    
    @app.get("/streams")
    async def list_streams():
        """List all active streams as JSON"""
        streams = rtmp_server.get_active_streams()
        return {"streams": streams}
    
    @app.get("/view/{stream_name}", response_class=HTMLResponse)
    async def view_stream(request: Request, stream_name: str):
        """Page to view a specific stream"""
        stream = rtmp_server.get_stream(stream_name)
        if not stream:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return templates.TemplateResponse(
            "view.html",
            {"request": request, "stream": stream}
        )
    
    @app.get("/thumbnails/{stream_name}.jpg")
    async def get_thumbnail(stream_name: str):
        """Get a thumbnail for a stream"""
        stream = rtmp_server.get_stream(stream_name)
        if not stream:
            raise HTTPException(status_code=404, detail="Stream not found")
            
        # Thumbnails would be generated in a real implementation
        # For now, we'll just return a placeholder
        thumbnail_path = static_dir / "img" / "placeholder.jpg"
        
        # If the thumbnail doesn't exist, return a default one
        if not thumbnail_path.exists():
            thumbnail_path = static_dir / "img" / "default-thumbnail.jpg"
            
        # If the default doesn't exist either, return a 404
        if not thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
            
        return FileResponse(str(thumbnail_path))
    
    # Demo endpoints for testing - these wouldn't be in a real application
    @app.post("/demo/add_stream")
    async def add_demo_stream(stream_name: str = Form(...)):
        """Add a demo stream (for testing only)"""
        if not stream_name:
            raise HTTPException(status_code=400, detail="Stream name is required")
            
        rtmp_url = f"rtmp://example.com/live/{stream_name}"
        stream = rtmp_server.add_stream(stream_name, rtmp_url)
        
        return RedirectResponse(url="/", status_code=303)
    
    @app.get("/demo/remove_stream/{stream_name}")
    async def remove_demo_stream(stream_name: str):
        """Remove a demo stream (for testing only)"""
        stream = rtmp_server.get_stream(stream_name)
        if not stream:
            raise HTTPException(status_code=404, detail="Stream not found")
            
        rtmp_server.remove_stream(stream_name)
        
        return RedirectResponse(url="/", status_code=303)
    
    # Shutdown event handler
    @app.on_event("shutdown")
    def shutdown_event():
        """Shutdown event handler"""
        rtmp_server.shutdown()
    
    return app