"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  FaceLandmarkerResult,
  HandLandmarkerResult,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

import { extractFrameFeatures, FEATURE_DIM } from "./features/featureExtractor";
import { VOCABULARY, DEFAULT_LABEL, DEFAULT_CLIPS_PER_LABEL, COUNTDOWN_SECONDS } from "./vocabulary";

const MODEL_BASE =
  "https://storage.googleapis.com/mediapipe-models/";

const HAND_MODEL_URL =
  MODEL_BASE + "hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL_URL =
  MODEL_BASE + "pose_landmarker/float16/1/pose_landmarker.task";
const FACE_MODEL_URL =
  MODEL_BASE + "face_landmarker/float16/1/face_landmarker.task";

interface RawLandmark {
  x: number;
  y: number;
  z: number;
}

interface RecordedClip {
  id: string;
  label: string;
  signerId: string;
  sessionId: string;
  frameCount: number;
  features: number[][];
  timestamp: number;
}

interface ClassStats {
  [label: string]: number;
}

interface QualityIssue {
  type: "no_hands" | "low_light" | "too_few_frames";
  message: string;
}

export const CollectPanel: React.FC = () => {
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [landmarksReady, setLandmarksReady] = useState<boolean>(false);

  // Collection state
  const [label, setLabel] = useState<string>(DEFAULT_LABEL);
  const [signerId, setSignerId] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>(
    "sess-" + Math.random().toString(36).substring(2, 9)
  );
  const [clipsPerLabel, setClipsPerLabel] = useState<number>(DEFAULT_CLIPS_PER_LABEL);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [countdown, setCountdown] = useState<number>(0);
  const [recordedClips, setRecordedClips] = useState<RecordedClip[]>([]);
  const [currentClipFrames, setCurrentClipFrames] = useState<number[][]>([]);
  const [classStats, setClassStats] = useState<ClassStats>({});
  const [qualityIssues, setQualityIssues] = useState<QualityIssue[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>("Ready");
  const [exporting, setExporting] = useState<boolean>(false);

  // MediaPipe refs
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize MediaPipe
  useEffect(() => {
    let mounted = true;
    const init = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        const [hands, pose, face] = await Promise.all([
          HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_URL },
            runningMode: "VIDEO",
            numHands: 2,
          }),
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL_URL },
            runningMode: "VIDEO",
          }),
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: FACE_MODEL_URL },
            runningMode: "VIDEO",
            numFaces: 1,
          }),
        ]);

        if (!mounted) return;
        handLandmarkerRef.current = hands;
        poseLandmarkerRef.current = pose;
        faceLandmarkerRef.current = face;
        setLandmarksReady(true);
        setStatusMessage("Models loaded. Start camera to begin.");
      } catch (err) {
        if (!mounted) return;
        console.error("MediaPipe init failed", err);
        setStatusMessage("MediaPipe models unavailable — running in demo mode.");
        setLandmarksReady(true);
      }
    };
    init();
    return () => {
      mounted = false;
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Connect WebSocket
  useEffect(() => {
    if (!("WebSocket" in window)) return;
    try {
      const ws = new WebSocket("ws://127.0.0.1:8000/stream/ws");
      ws.onopen = () => {
        wsRef.current = ws;
        setStatusMessage((prev) => prev + " | WebSocket connected");
      };
      ws.onclose = () => {
        wsRef.current = null;
        setStatusMessage("WebSocket disconnected");
      };
      ws.onerror = () => {
        wsRef.current = null;
      };
      return () => {
        ws.close();
      };
    } catch {
      // ignore
    }
  }, []);

  // Start/stop camera
  const toggleCamera = useCallback(async () => {
    if (isCameraActive) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      setIsCameraActive(false);
      setStatusMessage("Camera stopped.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          videoRef.current!.onloadedmetadata = () => resolve(void 0);
        });
        videoRef.current.play();
      }
      setIsCameraActive(true);
      setCameraError(null);
      setStatusMessage("Camera active. Detecting landmarks...");
      requestAnimationFrame(loop);
    } catch {
      setCameraError("Camera access denied or unavailable.");
      setStatusMessage("Camera error — using demo mode.");
    }
  }, [isCameraActive]);

  const loop = useCallback((time: number) => {
    if (!isCameraActive) return;
    if (!videoRef.current) return;

    const delta = time - lastTimeRef.current;
    lastTimeRef.current = time;
    frameCountRef.current += 1;
    if (delta >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastTimeRef.current = time;
    }

    if (
      landmarksReady &&
      (handLandmarkerRef.current || poseLandmarkerRef.current || faceLandmarkerRef.current) &&
      videoRef.current.readyState >= 2
    ) {
      detectAndDraw(videoRef.current);
    }

    animationFrameRef.current = requestAnimationFrame(loop);
  }, [isCameraActive, landmarksReady]);

  useEffect(() => {
    if (isCameraActive) {
      lastTimeRef.current = performance.now();
      animationFrameRef.current = requestAnimationFrame(loop);
    }
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isCameraActive, loop]);

  const detectAndDraw = useCallback(
    (video: HTMLVideoElement) => {
      const nowMs = performance.now();
      const results: {
        leftHand: RawLandmark[] | null;
        rightHand: RawLandmark[] | null;
        pose: RawLandmark[] | null;
        face: RawLandmark[] | null;
      } = { leftHand: null, rightHand: null, pose: null, face: null };

      try {
        if (handLandmarkerRef.current) {
          const handResult = handLandmarkerRef.current.detectForVideo(
            video,
            nowMs
          ) as HandLandmarkerResult;
          const left: RawLandmark[] = [];
          const right: RawLandmark[] = [];
          for (let i = 0; i < handResult.landmarks.length; i++) {
            const landmarks = handResult.landmarks[i];
            const handedness = handResult.handednesses[i]?.[0]?.categoryName;
            const pts: RawLandmark[] = landmarks.map((lm) => ({
              x: lm.x,
              y: lm.y,
              z: lm.z,
            }));
            if (handedness === "Left") left.push(...pts);
            else right.push(...pts);
          }
          results.leftHand = left.length > 0 ? left : null;
          results.rightHand = right.length > 0 ? right : null;
        }

        if (poseLandmarkerRef.current) {
          const poseResult = poseLandmarkerRef.current.detectForVideo(
            video,
            nowMs
          ) as PoseLandmarkerResult;
          if (poseResult.landmarks[0]) {
            results.pose = poseResult.landmarks[0].map((lm) => ({
              x: lm.x,
              y: lm.y,
              z: lm.z,
            }));
          }
        }

        if (faceLandmarkerRef.current) {
          const faceResult = faceLandmarkerRef.current.detectForVideo(
            video,
            nowMs
          ) as FaceLandmarkerResult;
          if (faceResult.landmarks[0]) {
            results.face = faceResult.landmarks[0].map((lm) => ({
              x: lm.x,
              y: lm.y,
              z: lm.z,
            }));
          }
        }
      } catch {
        // detection may fail on some frames; keep nulls
      }

      drawOverlay(results, video.videoWidth, video.videoHeight);

      // Quality checks
      const issues: QualityIssue[] = [];
      if (!results.leftHand && !results.rightHand) {
        issues.push({
          type: "no_hands",
          message: "No hands detected — move hands into frame.",
        });
      }
      if (issues.length > 0) {
        setQualityIssues(issues);
      } else {
        setQualityIssues([]);
      }

      // Stream to backend via WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && isRecording) {
        const framePayload = {
          left_hand: results.leftHand,
          right_hand: results.rightHand,
          pose: results.pose,
          face: results.face,
        };
        try {
          const features = extractFrameFeatures(framePayload as any);
          wsRef.current.send(
            JSON.stringify({
              session_id: sessionId,
              frames: [framePayload],
            })
          );
          setCurrentClipFrames((prev) => [...prev, features]);
        } catch {
          // ignore feature errors during streaming
        }
      }
    },
    [isRecording, sessionId]
  );

  const drawOverlay = useCallback(
    (results: {
      leftHand: RawLandmark[] | null;
      rightHand: RawLandmark[] | null;
      pose: RawLandmark[] | null;
      face: RawLandmark[] | null;
    },
    width: number,
    height: number
    ) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) return;
      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, width, height);

      const drawPoints = (pts: RawLandmark[] | null, color: string) => {
        if (!pts) return;
        ctx.fillStyle = color;
        for (const lm of pts) {
          ctx.beginPath();
          ctx.arc(lm.x * width, lm.y * height, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      };

      drawPoints(results.leftHand, "#06b6d4");
      drawPoints(results.rightHand, "#3b82f6");
      drawPoints(results.pose, "#10b981");
      drawPoints(results.face, "#a855f7");
    },
    []
  );

  // Countdown + recording
  useEffect(() => {
    if (!isRecording || countdown <= 0) return;
    const timer = setTimeout(() => {
      setCountdown((c) => (c > 1 ? c - 1 : 0));
    }, 1000);
    return () => clearTimeout(timer);
  }, [isRecording, countdown]);

  useEffect(() => {
    if (countdown === 0 && isRecording) {
      setIsRecording(false);
      setStatusMessage(`Captured clip for "${label}".`);
      // Save current clip
      if (currentClipFrames.length > 0) {
        const clip: RecordedClip = {
          id: Date.now().toString(),
          label,
          signerId,
          sessionId,
          frameCount: currentClipFrames.length,
          features: currentClipFrames,
          timestamp: Date.now(),
        };
        setRecordedClips((prev) => [...prev, clip]);
        setClassStats((prev) => ({
          ...prev,
          [label]: (prev[label] || 0) + 1,
        }));
      }
      setCurrentClipFrames([]);
    }
  }, [countdown, isRecording, label, signerId, sessionId, currentClipFrames]);

  const startRecording = useCallback(() => {
    if (!label || !signerId.trim()) {
      setStatusMessage("Please choose a label and enter a signer ID.");
      return;
    }
    setCurrentClipFrames([]);
    setCountdown(COUNTDOWN_SECONDS);
    setIsRecording(true);
    setStatusMessage(`Get ready — recording "${label}" in ${COUNTDOWN_SECONDS}...`);
  }, [label, signerId]);

  const deleteClip = useCallback((id: string) => {
    setRecordedClips((prev) => {
      const removed = prev.find((c) => c.id === id);
      const next = prev.filter((c) => c.id !== id);
      if (removed) {
        setClassStats((st) => {
          const n = Math.max(0, (st[removed.label] || 1) - 1);
          return { ...st, [removed.label]: n };
        });
      }
      return next;
    });
  }, []);

  const clearAllClips = useCallback(() => {
    setRecordedClips([]);
    setClassStats({});
  }, []);

  const fetchClassStats = useCallback(async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/collect/stats");
      if (res.ok) {
        const data = await res.json();
        setClassStats(data.labels || {});
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    fetchClassStats();
  }, [fetchClassStats]);

  const handleExport = useCallback(async () => {
    if (recordedClips.length === 0) {
      setStatusMessage("No clips to export.");
      return;
    }
    setExporting(true);
    try {
      const exportData = {
        format: "signbridge-collect-v1",
        signerId,
        sessionId,
        featureDim: FEATURE_DIM,
        clips: recordedClips.map((c) => ({
          id: c.id,
          label: c.label,
          frameCount: c.frameCount,
          features: c.features,
          timestamp: c.timestamp,
        })),
      };
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `signbridge_collect_${sessionId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setStatusMessage(`Exported ${recordedClips.length} clips to JSON.`);
    } catch {
      setStatusMessage("Export failed.");
    } finally {
      setExporting(false);
    }
  }, [recordedClips, signerId, sessionId]);

  const clipsForLabel = recordedClips.filter((c) => c.label === label);

  return (
    <div className="collect-panel">
      <div className="collect-header">
        <h2 className="collect-title">📊 Data Collection — /collect</h2>
        <span className={`status-badge ${isCameraActive ? "bg-success" : "bg-muted"}`}>
          {isCameraActive ? "Camera: ON" : "Camera: OFF"}
        </span>
        <span className={`status-badge ${landmarksReady ? "bg-success" : "bg-muted"}`}>
          {landmarksReady ? "MediaPipe: ready" : "MediaPipe: loading…"}
        </span>
        <span className="status-badge bg-info">FPS: {fps}</span>
      </div>

      {cameraError && (
        <div className="error-banner">{cameraError}</div>
      )}

      <div className="collect-grid">
        {/* Left: Camera + Controls */}
        <div className="collect-left">
          <div className="camera-card">
            <div className="camera-wrapper">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="camera-video"
              />
              <canvas ref={canvasRef} className="camera-overlay" />
              {!isCameraActive && (
                <div className="camera-placeholder">
                  <p>Camera off. Click "Start Camera" to begin.</p>
                </div>
              )}
            </div>

            <div className="camera-actions">
              <button
                className={`action-btn ${isCameraActive ? "btn-stop" : "btn-start"}`}
                onClick={toggleCamera}
                disabled={!landmarksReady && !cameraError}
              >
                {isCameraActive ? "⏹ Stop Camera" : "▶ Start Camera"}
              </button>
            </div>

            <div className="fps-row">
              <span className="fps-label">FPS:</span>
              <span className="fps-value">{fps}</span>
              <span className="fps-label">Feature dim:</span>
              <span className="fps-value">{FEATURE_DIM}</span>
            </div>

            {qualityIssues.length > 0 && (
              <div className="quality-warnings">
                {qualityIssues.map((issue, i) => (
                  <div key={i} className="quality-warning">{issue.message}</div>
                ))}
              </div>
            )}
          </div>

          <div className="control-card">
            <div className="form-row">
              <label className="form-label">Sign label</label>
              <select
                className="form-select"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                disabled={isRecording}
              >
                {VOCABULARY.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-row">
              <label className="form-label">Signer ID</label>
              <input
                className="form-input"
                type="text"
                value={signerId}
                onChange={(e) => setSignerId(e.target.value)}
                placeholder="e.g. signer_01"
                disabled={isRecording}
              />
            </div>

            <div className="form-row">
              <label className="form-label">Session ID</label>
              <input
                className="form-input"
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                disabled={isRecording}
              />
            </div>

            <div className="form-row">
              <label className="form-label">Clips per label</label>
              <input
                className="form-input"
                type="number"
                value={clipsPerLabel}
                onChange={(e) => setClipsPerLabel(Math.max(1, parseInt(e.target.value) || 1))}
                min="1"
                disabled={isRecording}
              />
            </div>

            <div className="form-row">
              <label className="form-label">Recording action</label>
              <button
                className={`action-btn ${isRecording ? "btn-recording" : "btn-record"}`}
                onClick={startRecording}
                disabled={isRecording || !signerId.trim() || !label}
              >
                {countdown > 0 ? `⏳ ${countdown}…` : "🔴 Record Clip"}
              </button>
            </div>

            <div className="status-line">{statusMessage}</div>

            <div className="clip-tray">
              <h3 className="tray-title">Clips for "{label}" ({clipsForLabel.length})</h3>
              {clipsForLabel.length === 0 ? (
                <p className="empty-tray">No clips recorded yet.</p>
              ) : (
                <div className="clip-list">
                  {clipsForLabel.map((c) => (
                    <div key={c.id} className="clip-item">
                      <span className="clip-meta">{c.frameCount} frames • {new Date(c.timestamp).toLocaleTimeString()}</span>
                      <button className="clip-delete" onClick={() => deleteClip(c.id)}>✕</button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Dashboard + Export */}
        <div className="collect-right">
          <div className="dashboard-card">
            <h3 className="dashboard-title">Per-Class Count Dashboard</h3>
            <div className="stats-grid">
              {VOCABULARY.map((l) => (
                <div key={l} className="stat-card">
                  <span className="stat-label">{l}</span>
                  <span className="stat-value">{classStats[l] || 0}</span>
                </div>
              ))}
            </div>
            <div className="dashboard-actions">
              <button className="action-btn btn-outline" onClick={fetchClassStats}>
                🔄 Refresh from server
              </button>
              <button className="action-btn btn-danger" onClick={clearAllClips}>
                🗑 Clear all local clips
              </button>
            </div>
          </div>

          <div className="export-card">
            <h3 className="dashboard-title">Export Dataset</h3>
            <p className="export-hint">
              Exports recorded clips as JSON to <code>ml/datasets/</code> (gitignored).
              Each clip contains {FEATURE_DIM}-dim feature vectors per frame.
            </p>
            <div className="export-actions">
              <button
                className="action-btn btn-export"
                onClick={handleExport}
                disabled={exporting || recordedClips.length === 0}
              >
                {exporting ? "Exporting…" : `⬇ Export ${recordedClips.length} clips`}
              </button>
            </div>
            <p className="export-hint">
              Total local clips: <strong>{recordedClips.length}</strong>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
