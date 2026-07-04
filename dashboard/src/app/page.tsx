"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause, Square, Crosshair, Map, ShieldAlert, History, Activity, AlertTriangle, Settings, Camera, Check, Trash2, Link2 } from "lucide-react";

export default function Dashboard() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [forensicMode, setForensicMode] = useState(false);
  const [isTrackingMode, setIsTrackingMode] = useState(false);
  const [isGeofenceMode, setIsGeofenceMode] = useState(false);
  const [geofencePoints, setGeofencePoints] = useState<[number, number][]>([]);
  const [geofenceCommitted, setGeofenceCommitted] = useState(false);
  const [trackingPoint, setTrackingPoint] = useState<[number, number] | null>(null);
  const [trackingBbox, setTrackingBbox] = useState<[number, number, number, number] | null>(null);
  const [trackingConfidence, setTrackingConfidence] = useState(0);
  const [samConnected, setSamConnected] = useState(false);
  const [frameData, setFrameData] = useState<string | null>(null);
  
  // Dynamic Colab Endpoint Configuration State
  const [ngrokUrl, setNgrokUrl] = useState("https://b2d9-34-6-88-71.ngrok-free.app");
  const [isApplyingUrl, setIsApplyingUrl] = useState(false);
  const [urlAppliedSuccess, setUrlAppliedSuccess] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load saved ngrok URL from local storage on mount
  useEffect(() => {
    const savedUrl = localStorage.getItem("chrono_ngrok_url");
    if (savedUrl) {
      setNgrokUrl(savedUrl);
    }
  }, []);

  // Callback to submit/apply the endpoint URL to backend
  const handleApplyNgrokUrl = useCallback(() => {
    localStorage.setItem("chrono_ngrok_url", ngrokUrl);
    setIsApplyingUrl(true);
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "update_endpoint", endpoint_url: ngrokUrl }));
      // We will also reset samConnected since we just changed target predictor
      setSamConnected(false);
      setTimeout(() => {
        setIsApplyingUrl(false);
      }, 600);
    } else {
      setIsApplyingUrl(false);
      setUrlAppliedSuccess(true);
      setTimeout(() => setUrlAppliedSuccess(false), 2000);
    }
  }, [ngrokUrl]);


  // The actual frame resolution from the backend
  const FRAME_W = 640;
  const FRAME_H = 480;

  // ------------------------------------------------------------------
  // Helper: Convert a click on the <img> (which uses object-contain)
  // to true pixel coordinates on the underlying 640×480 frame.
  // ------------------------------------------------------------------
  const clientToFrameCoords = useCallback((clientX: number, clientY: number): [number, number] | null => {
    const img = imgRef.current;
    if (!img) return null;

    const rect = img.getBoundingClientRect();

    // Compute the rendered size & offset caused by object-contain
    const imgAspect = FRAME_W / FRAME_H;
    const boxAspect = rect.width / rect.height;

    let renderedW: number, renderedH: number, offsetX: number, offsetY: number;

    if (boxAspect > imgAspect) {
      // Letterboxed: bars on left/right
      renderedH = rect.height;
      renderedW = rect.height * imgAspect;
      offsetX = (rect.width - renderedW) / 2;
      offsetY = 0;
    } else {
      // Pillarboxed: bars on top/bottom
      renderedW = rect.width;
      renderedH = rect.width / imgAspect;
      offsetX = 0;
      offsetY = (rect.height - renderedH) / 2;
    }

    // Position relative to the rendered image area
    const relX = clientX - rect.left - offsetX;
    const relY = clientY - rect.top - offsetY;

    // Reject clicks in the padding (letterbox/pillarbox bars)
    if (relX < 0 || relY < 0 || relX > renderedW || relY > renderedH) {
      return null;
    }

    const frameX = Math.round((relX / renderedW) * FRAME_W);
    const frameY = Math.round((relY / renderedH) * FRAME_H);

    return [
      Math.max(0, Math.min(FRAME_W - 1, frameX)),
      Math.max(0, Math.min(FRAME_H - 1, frameY)),
    ];
  }, []);

  // ------------------------------------------------------------------
  // Helper: Convert frame coords back to canvas-pixel coords for
  // drawing the overlay (geofence polygon, tracking crosshair, etc.)
  // ------------------------------------------------------------------
  const frameToCanvasCoords = useCallback((fx: number, fy: number): [number, number] | null => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return null;

    const rect = img.getBoundingClientRect();
    const imgAspect = FRAME_W / FRAME_H;
    const boxAspect = rect.width / rect.height;

    let renderedW: number, renderedH: number, offsetX: number, offsetY: number;

    if (boxAspect > imgAspect) {
      renderedH = rect.height;
      renderedW = rect.height * imgAspect;
      offsetX = (rect.width - renderedW) / 2;
      offsetY = 0;
    } else {
      renderedW = rect.width;
      renderedH = rect.width / imgAspect;
      offsetX = 0;
      offsetY = (rect.height - renderedH) / 2;
    }

    const cx = offsetX + (fx / FRAME_W) * renderedW;
    const cy = offsetY + (fy / FRAME_H) * renderedH;
    return [cx, cy];
  }, []);

  // ------------------------------------------------------------------
  // Canvas overlay – redraws geofence polygon & tracking crosshair
  // on top of the video <img> every frame.
  // ------------------------------------------------------------------
  useEffect(() => {
    let raf: number;
    const draw = () => {
      const canvas = canvasRef.current;
      const img = imgRef.current;
      if (!canvas || !img) {
        raf = requestAnimationFrame(draw);
        return;
      }

      // Sync canvas pixel size to the image element's layout size
      const rect = img.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) { raf = requestAnimationFrame(draw); return; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // --- Draw geofence polygon ---
      if (geofencePoints.length > 0) {
        const pts = geofencePoints.map(([fx, fy]) => frameToCanvasCoords(fx, fy)).filter(Boolean) as [number, number][];

        if (pts.length > 0) {
          // Draw filled semi-transparent area if committed (>=3 points)
          if (geofenceCommitted && pts.length >= 3) {
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
            ctx.closePath();
            ctx.fillStyle = "rgba(0, 240, 255, 0.08)";
            ctx.fill();
          }

          // Draw lines
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
          if (geofenceCommitted) ctx.closePath();
          ctx.strokeStyle = "rgba(0, 240, 255, 0.9)";
          ctx.lineWidth = 2;
          ctx.setLineDash(geofenceCommitted ? [] : [6, 4]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw vertices
          for (const [cx, cy] of pts) {
            ctx.beginPath();
            ctx.arc(cx, cy, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#00f0ff";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(cx, cy, 7, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(0, 240, 255, 0.5)";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // --- Draw tracking bounding box (from SAM 2) ---
      if (trackingBbox) {
        const [bx, by, bw, bh] = trackingBbox;
        const topLeft = frameToCanvasCoords(bx, by);
        const bottomRight = frameToCanvasCoords(bx + bw, by + bh);

        if (topLeft && bottomRight) {
          const [x1, y1] = topLeft;
          const [x2, y2] = bottomRight;
          const w = x2 - x1;
          const h = y2 - y1;

          // Glow effect
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 8;

          // Bounding box
          ctx.strokeStyle = "rgba(0, 240, 255, 0.9)";
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, w, h);

          // Corner brackets for premium look
          const bracketLen = Math.min(w, h) * 0.2;
          ctx.lineWidth = 3;
          ctx.strokeStyle = "#00f0ff";
          // Top-left
          ctx.beginPath();
          ctx.moveTo(x1, y1 + bracketLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + bracketLen, y1);
          ctx.stroke();
          // Top-right
          ctx.beginPath();
          ctx.moveTo(x2 - bracketLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + bracketLen);
          ctx.stroke();
          // Bottom-left
          ctx.beginPath();
          ctx.moveTo(x1, y2 - bracketLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + bracketLen, y2);
          ctx.stroke();
          // Bottom-right
          ctx.beginPath();
          ctx.moveTo(x2 - bracketLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - bracketLen);
          ctx.stroke();

          ctx.shadowBlur = 0;

          // Semi-transparent fill
          ctx.fillStyle = "rgba(0, 240, 255, 0.05)";
          ctx.fillRect(x1, y1, w, h);

          // Centroid crosshair inside the bbox
          if (trackingPoint) {
            const cp = frameToCanvasCoords(trackingPoint[0], trackingPoint[1]);
            if (cp) {
              const [ccx, ccy] = cp;
              ctx.strokeStyle = "rgba(0, 240, 255, 0.6)";
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(ccx - 8, ccy); ctx.lineTo(ccx + 8, ccy);
              ctx.moveTo(ccx, ccy - 8); ctx.lineTo(ccx, ccy + 8);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }

          // Label with confidence
          const confPct = Math.round(trackingConfidence * 100);
          const confColor = confPct > 70 ? "#00ff88" : confPct > 40 ? "#ffaa00" : "#ff4444";
          ctx.font = "bold 11px monospace";
          ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
          ctx.fillRect(x1, y1 - 20, 180, 18);
          ctx.fillStyle = "#00f0ff";
          ctx.fillText("SAM3 TRACKING", x1 + 4, y1 - 6);
          ctx.fillStyle = confColor;
          ctx.fillText(`${confPct}%`, x1 + 120, y1 - 6);

          // Confidence bar
          const barW = 50;
          const barH = 3;
          const barX = x1 + 140;
          const barY = y1 - 10;
          ctx.fillStyle = "rgba(255,255,255,0.15)";
          ctx.fillRect(barX, barY, barW, barH);
          ctx.fillStyle = confColor;
          ctx.fillRect(barX, barY, barW * trackingConfidence, barH);
        }
      } else if (trackingPoint) {
        // Fallback: just a crosshair at the click point (SAM not connected)
        const tp = frameToCanvasCoords(trackingPoint[0], trackingPoint[1]);
        if (tp) {
          const [cx, cy] = tp;
          const size = 18;
          const gap = 5;

          ctx.strokeStyle = "#00f0ff";
          ctx.lineWidth = 2;
          ctx.shadowColor = "#00f0ff";
          ctx.shadowBlur = 6;

          ctx.beginPath();
          ctx.moveTo(cx - size, cy); ctx.lineTo(cx - gap, cy);
          ctx.moveTo(cx + gap, cy); ctx.lineTo(cx + size, cy);
          ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy - gap);
          ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, cy + size);
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(cx, cy, size + 2, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
          ctx.lineWidth = 1;
          ctx.stroke();

          ctx.shadowBlur = 0;
          ctx.font = "11px monospace";
          ctx.fillStyle = "#ffaa00";
          ctx.fillText("TRACKING (SAM3 offline)", cx + size + 6, cy - 4);
          ctx.shadowBlur = 0;
        }
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [geofencePoints, geofenceCommitted, trackingPoint, trackingBbox, trackingConfidence, frameToCanvasCoords]);

  // ------------------------------------------------------------------
  // Click handler for the video canvas overlay
  // ------------------------------------------------------------------
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const result = clientToFrameCoords(e.clientX, e.clientY);
    if (!result) return; // click was in padding
    const [x, y] = result;

    if (isTrackingMode) {
      ws.send(JSON.stringify({ type: "init_tracking", x, y }));
      setTrackingPoint([x, y]);
      setIsTrackingMode(false);
    } else if (isGeofenceMode && !geofenceCommitted) {
      const newPoints: [number, number][] = [...geofencePoints, [x, y]];
      setGeofencePoints(newPoints);
      // Don't send to backend until committed
    }
  }, [isTrackingMode, isGeofenceMode, geofencePoints, geofenceCommitted, clientToFrameCoords]);

  // ------------------------------------------------------------------
  // Commit geofence: send finalized polygon to backend
  // ------------------------------------------------------------------
  const commitGeofence = useCallback(() => {
    if (geofencePoints.length < 3) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Send the points as an array of [x, y] to the backend
    ws.send(JSON.stringify({ type: "set_geofence", points: geofencePoints.map(([x, y]) => [x, y]) }));
    setGeofenceCommitted(true);
    setIsGeofenceMode(false);
  }, [geofencePoints]);

  // ------------------------------------------------------------------
  // Clear geofence
  // ------------------------------------------------------------------
  const clearGeofence = useCallback(() => {
    setGeofencePoints([]);
    setGeofenceCommitted(false);
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "set_geofence", points: [] }));
    }
  }, []);

  // WebSocket Connection Logic
  useEffect(() => {
    if (isStreaming) {
      wsRef.current = new WebSocket("ws://localhost:8000/ws/video");
      
      wsRef.current.onopen = () => {
        const savedUrl = localStorage.getItem("chrono_ngrok_url") || ngrokUrl;
        if (savedUrl && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "update_endpoint", endpoint_url: savedUrl }));
        }
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "frame") {
          setFrameData(data.data);
        } else if (data.type === "tracking_update") {
          // Real-time tracking data from SAM 2 via backend
          if (data.bbox) {
            setTrackingBbox(data.bbox as [number, number, number, number]);
            setSamConnected(true);
          }
          if (data.centroid) {
            setTrackingPoint(data.centroid as [number, number]);
          }
          setTrackingConfidence(data.confidence ?? 0);
        } else if (data.type === "alert") {
          // Real-time alert from backend (e.g. geofence breach)
          setAlerts((prev) => [data.data, ...prev]);
        } else if (data.type === "endpoint_updated") {
          setUrlAppliedSuccess(true);
          setTimeout(() => setUrlAppliedSuccess(false), 3000);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket Error:", error);
      };
    } else {
      if (wsRef.current) {
        wsRef.current.close();
      }
      setFrameData(null);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isStreaming, ngrokUrl]);

  const [alerts, setAlerts] = useState<{ id: number; time: string; type: string }[]>([
    { id: 1, time: "10:42:05", type: "Geofence Breach" },
    { id: 2, time: "10:38:12", type: "Object Missing" },
  ]);

  return (
    <div className="min-h-screen p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative overflow-hidden">
      {/* Background glow effects */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-accent/20 blur-[120px] rounded-full pointer-events-none" />

      {/* Sidebar - Controls */}
      <div className="lg:col-span-3 flex flex-col gap-6 z-10">
        <div className="glass-panel p-6">
          <h1 className="text-2xl font-bold tracking-wider flex items-center gap-3 mb-1">
            <Activity className="text-primary" /> 
            <span className="neon-text text-primary">CHRONO</span>GUARD
          </h1>
          <p className="text-sm text-muted mb-6">Agentic Video Forensics</p>

          <div className="space-y-4">
            <h2 className="text-xs uppercase tracking-widest text-muted font-semibold mb-2">System Status</h2>
            
            <div className="flex items-center justify-between p-3 rounded-lg bg-black/40 border border-border">
              <span className="flex items-center gap-2 text-sm"><div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} /> Backend</span>
              <span className="text-xs font-mono text-muted">{isStreaming ? 'CONNECTED' : 'OFFLINE'}</span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-black/40 border border-border">
              <span className="flex items-center gap-2 text-sm"><div className={`w-2 h-2 rounded-full ${samConnected ? 'bg-green-500 animate-pulse' : isStreaming ? 'bg-yellow-500' : 'bg-red-500'}`} /> SAM 3 Predictor</span>
              <span className="text-xs font-mono text-muted">{samConnected ? 'TRACKING' : trackingPoint ? 'OFFLINE' : 'IDLE'}</span>
            </div>
          </div>

          <div className="border-t border-border/40 my-5" />

          <div className="space-y-3">
            <h2 className="text-xs uppercase tracking-widest text-muted font-semibold flex items-center justify-between">
              <span>SAM 3 Colab Bridge</span>
              <Link2 className="w-3.5 h-3.5 text-primary animate-pulse" />
            </h2>
            <p className="text-[10.5px] text-muted leading-relaxed">
              Route vision tasks to your hosted SAM 3 engine. Paste the public ngrok endpoint URL from Colab:
            </p>
            <div className="relative group">
              <input
                type="text"
                value={ngrokUrl}
                onChange={(e) => setNgrokUrl(e.target.value)}
                placeholder="https://your-tunnel.ngrok-free.app"
                className="w-full bg-black/55 border border-white/10 rounded-lg p-2.5 pl-3 pr-20 text-[11px] font-mono text-white placeholder-white/20 transition-all duration-300 focus:outline-none focus:border-primary/80 focus:ring-1 focus:ring-primary/20"
              />
              <button
                onClick={handleApplyNgrokUrl}
                disabled={isApplyingUrl}
                className="absolute right-1 top-1 bottom-1 px-3 text-[10px] font-medium bg-primary/10 text-primary border border-primary/25 rounded-md hover:bg-primary/20 active:scale-95 transition-all flex items-center gap-1 cursor-pointer disabled:opacity-50"
              >
                {urlAppliedSuccess ? (
                  <>
                    <Check className="w-3 h-3 text-green-400" />
                    <span className="text-green-400">Linked</span>
                  </>
                ) : isApplyingUrl ? (
                  <span>Syncing...</span>
                ) : (
                  <span>Apply</span>
                )}
              </button>
            </div>
            {urlAppliedSuccess && (
              <p className="text-[10px] text-green-400 flex items-center gap-1 animate-fadeIn">
                <Check className="w-2.5 h-2.5" /> Bridge synchronized!
              </p>
            )}
          </div>
        </div>

        <div className="glass-panel p-6 flex-1">
          <h2 className="text-xs uppercase tracking-widest text-muted font-semibold mb-4">Command Center</h2>
          
          <div className="grid grid-cols-2 gap-3 mb-6">
            <button 
              onClick={() => setIsStreaming(!isStreaming)}
              className={`glass-button p-4 flex flex-col items-center justify-center gap-2 ${isStreaming ? 'border-primary/50 bg-primary/10' : ''}`}
            >
              {isStreaming ? <Pause className="text-primary" /> : <Play className="text-primary" />}
              <span className="text-xs">{isStreaming ? 'Pause Stream' : 'Start Stream'}</span>
            </button>
            <button 
              onClick={() => {
                if (wsRef.current) wsRef.current.send(JSON.stringify({ type: "stop_analysis" }));
                setIsTrackingMode(false);
                setIsGeofenceMode(false);
                setGeofencePoints([]);
                setGeofenceCommitted(false);
                setTrackingPoint(null);
                setTrackingBbox(null);
                setTrackingConfidence(0);
                setSamConnected(false);
              }}
              className="glass-button p-4 flex flex-col items-center justify-center gap-2 hover:border-accent/50 hover:bg-accent/10 transition-colors group">
              <Square className="text-muted group-hover:text-accent transition-colors" />
              <span className="text-xs">Stop Analysis</span>
            </button>
          </div>

          <div className="space-y-3">
            <button 
              onClick={() => { setIsTrackingMode(!isTrackingMode); setIsGeofenceMode(false); }}
              className={`w-full glass-button p-3 flex items-center justify-between group ${isTrackingMode ? 'border-primary/50 bg-primary/10' : ''}`}>
              <div className="flex items-center gap-3">
                <Crosshair className={`w-4 h-4 transition-colors ${isTrackingMode ? 'text-primary' : 'text-muted group-hover:text-primary'}`} />
                <span className="text-sm">Initialize Tracking (Click)</span>
              </div>
              {trackingPoint && !isTrackingMode && (
                <span className={`text-[10px] font-mono ${samConnected ? 'text-green-400' : 'text-yellow-400'}`}>
                  {samConnected ? 'SAM3 LIVE' : 'FALLBACK'}
                </span>
              )}
            </button>
            <button 
              onClick={() => {
                if (isGeofenceMode) {
                  // Clicking again while in geofence mode = cancel
                  setIsGeofenceMode(false);
                } else {
                  setIsGeofenceMode(true);
                  setIsTrackingMode(false);
                  if (geofenceCommitted) {
                    // Clear old geofence when re-entering draw mode
                    clearGeofence();
                  }
                }
              }}
              className={`w-full glass-button p-3 flex items-center justify-between group ${isGeofenceMode ? 'border-primary/50 bg-primary/10' : ''}`}>
              <div className="flex items-center gap-3">
                <Map className={`w-4 h-4 transition-colors ${isGeofenceMode ? 'text-primary' : 'text-muted group-hover:text-primary'}`} />
                <span className="text-sm">Draw Geofence</span>
              </div>
              {geofenceCommitted && !isGeofenceMode && (
                <span className="text-[10px] font-mono text-primary">SET</span>
              )}
            </button>

            {/* Geofence sub-controls: shown when in geofence drawing mode */}
            {isGeofenceMode && (
              <div className="pl-4 space-y-2 animate-fadeIn">
                <p className="text-[11px] text-muted">
                  Click on the video to place vertices. Min 3 points.
                </p>
                <p className="text-[11px] font-mono text-primary/70">
                  Points: {geofencePoints.length}
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={commitGeofence}
                    disabled={geofencePoints.length < 3}
                    className="flex-1 glass-button p-2 flex items-center justify-center gap-2 text-xs disabled:opacity-30 disabled:cursor-not-allowed hover:border-green-500/50 hover:bg-green-500/10 transition-colors"
                  >
                    <Check className="w-3 h-3" /> Commit
                  </button>
                  <button
                    onClick={clearGeofence}
                    className="flex-1 glass-button p-2 flex items-center justify-center gap-2 text-xs hover:border-accent/50 hover:bg-accent/10 transition-colors"
                  >
                    <Trash2 className="w-3 h-3" /> Clear
                  </button>
                </div>
              </div>
            )}

            <button 
              onClick={() => setForensicMode(!forensicMode)}
              className={`w-full glass-button p-3 flex items-center justify-between ${forensicMode ? 'border-primary/50 bg-primary/10' : ''}`}
            >
              <div className="flex items-center gap-3">
                <History className={`w-4 h-4 ${forensicMode ? 'text-primary' : 'text-muted'}`} />
                <span className="text-sm">Time-Hopping Forensics</span>
              </div>
              <div className={`w-8 h-4 rounded-full transition-colors flex items-center px-1 ${forensicMode ? 'bg-primary' : 'bg-black/50 border border-border'}`}>
                <div className={`w-2 h-2 rounded-full bg-white transition-transform ${forensicMode ? 'translate-x-4' : 'translate-x-0'}`} />
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Main Video View */}
      <div className="lg:col-span-6 flex flex-col z-10 h-[85vh] lg:h-auto">
        <div className="glass-panel p-2 flex-1 flex flex-col relative group">
          {/* Top Bar inside Video View */}
          <div className="absolute top-4 left-4 right-4 flex justify-between items-center z-20 pointer-events-none">
            <div className="px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-xs font-mono flex items-center gap-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              LIVE <span className="text-muted ml-2">FPS: --</span>
            </div>
            <div className="px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-xs flex items-center gap-2">
              <ShieldAlert className="w-3 h-3 text-primary" />
              Privacy Filter: ON
            </div>
          </div>

          {/* Mode indicator overlay */}
          {(isTrackingMode || isGeofenceMode) && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
              <div className="px-4 py-2 rounded-full bg-primary/20 backdrop-blur-md border border-primary/40 text-xs font-mono text-primary flex items-center gap-2 animate-pulse">
                {isTrackingMode && <><Crosshair className="w-3 h-3" /> Click an object to track</>}
                {isGeofenceMode && <><Map className="w-3 h-3" /> Click to place geofence vertex ({geofencePoints.length} placed)</>}
              </div>
            </div>
          )}

          {/* Video + Canvas Overlay */}
          <div ref={containerRef} className="w-full h-full bg-black/80 rounded-xl border border-white/5 flex items-center justify-center overflow-hidden relative">
            <Camera className="w-16 h-16 text-white/5" />
            
            {isStreaming && frameData ? (
              <>
                <img 
                  ref={imgRef}
                  src={`data:image/jpeg;base64,${frameData}`} 
                  alt="Live Stream" 
                  className="w-full h-full object-contain pointer-events-none select-none"
                  draggable={false}
                />
                {/* Canvas overlay – sits directly on top of the <img>, same size */}
                <canvas
                  ref={canvasRef}
                  className={`absolute inset-0 w-full h-full ${isTrackingMode || isGeofenceMode ? 'cursor-crosshair' : 'cursor-default'}`}
                  onClick={handleCanvasClick}
                />
              </>
            ) : isStreaming ? (
              <div className="absolute inset-0 flex items-center justify-center flex-col text-muted gap-4">
                <span className="text-sm tracking-widest uppercase animate-pulse">Connecting to backend...</span>
              </div>
            ) : null}
            
            {!isStreaming && (
              <div className="absolute inset-0 flex items-center justify-center flex-col text-muted gap-4">
                <span className="text-sm tracking-widest uppercase">Waiting for feed...</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right Sidebar - Alerts & Telemetry */}
      <div className="lg:col-span-3 flex flex-col gap-6 z-10">
        <div className="glass-panel p-6 flex-1 flex flex-col">
          <h2 className="text-xs uppercase tracking-widest text-muted font-semibold mb-4 flex items-center justify-between">
            <span>Alert Log</span>
            <AlertTriangle className="w-4 h-4 text-accent" />
          </h2>
          
          <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
            {alerts.map((alert) => (
              <div key={alert.id} className="p-3 rounded-lg bg-black/40 border border-accent/20 hover:border-accent/50 transition-colors relative overflow-hidden group">
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-accent group-hover:w-1.5 transition-all" />
                <div className="flex justify-between items-start mb-2 pl-2">
                  <span className="text-sm font-semibold text-white">{alert.type}</span>
                  <span className="text-xs font-mono text-muted">{alert.time}</span>
                </div>
                <div className="pl-2 flex items-center gap-2">
                  <div className="w-16 h-10 bg-white/5 rounded border border-white/10 flex items-center justify-center">
                    <Camera className="w-4 h-4 text-muted" />
                  </div>
                  <button className="text-xs text-primary hover:underline ml-auto">View Snippet</button>
                </div>
              </div>
            ))}
            
            <div className="p-4 border border-dashed border-white/10 rounded-lg text-center text-xs text-muted flex flex-col items-center gap-2">
              <Activity className="w-4 h-4 opacity-50" />
              Monitoring active...
            </div>
          </div>
        </div>

        <div className="glass-panel p-6">
          <h2 className="text-xs uppercase tracking-widest text-muted font-semibold mb-4">Telemetry</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-muted">Network Latency</span>
                <span className="font-mono text-primary">~12ms</span>
              </div>
              <div className="w-full h-1 bg-black/50 rounded-full overflow-hidden">
                <div className="w-1/4 h-full bg-primary" />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-muted">Colab GPU VRAM</span>
                <span className="font-mono text-accent">--%</span>
              </div>
              <div className="w-full h-1 bg-black/50 rounded-full overflow-hidden">
                <div className="w-0 h-full bg-accent" />
              </div>
            </div>
          </div>
        </div>
      </div>
      
    </div>
  );
}
