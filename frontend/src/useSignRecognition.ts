"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  FaceLandmarkerResult,
  HandLandmarkerResult,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

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

interface RecognizedTokenEvent {
  token: string;
  confidence: number;
  timestamp_ms?: number;
}

interface UseSignRecognitionProps {
  sessionId: string;
  onToken: (token: RecognizedTokenEvent) => void;
  onStatus?: (message: string) => void;
}

export function useSignRecognition({
  sessionId,
  onToken,
  onStatus,
}: UseSignRecognitionProps) {
  const [isActive, setIsActive] = useState<boolean>(false);
  const [ready, setReady] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);

  const handRef = useRef<HandLandmarker | null>(null);
  const poseRef = useRef<PoseLandmarker | null>(null);
  const faceRef = useRef<FaceLandmarker | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const countRef = useRef<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Init MediaPipe
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
        handRef.current = hands;
        poseRef.current = pose;
        faceRef.current = face;
        setReady(true);
        onStatus?.("Recognition models loaded.");
      } catch (err) {
        if (!mounted) return;
        console.error("MediaPipe init failed", err);
        setError("MediaPipe models unavailable — recognition in demo mode.");
        setReady(true);
        onStatus?.("MediaPipe unavailable — demo mode.");
      }
    };
    init();
    return () => {
      mounted = false;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [onStatus]);

  // WebSocket
  useEffect(() => {
    if (!("WebSocket" in window)) return;
    try {
      const ws = new WebSocket("ws://127.0.0.1:8000/stream/ws");
      ws.onopen = () => {
        wsRef.current = ws;
        onStatus?.("Recognition stream connected.");
      };
      ws.onclose = () => {
        wsRef.current = null;
        onStatus?.("Recognition stream disconnected.");
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.token) {
            onToken({
              token: String(msg.token.token),
              confidence: Number(msg.token.confidence) || 0,
              timestamp_ms: msg.token.timestamp_ms,
            });
          }
        } catch {
          // ignore malformed messages
        }
      };
      ws.onerror = () => {
        wsRef.current = null;
      };
      return () => ws.close();
    } catch {
      // ignore
    }
  }, [onToken, onStatus]);

  const detect = useCallback(
    (video: HTMLVideoElement) => {
      if (
        !ready ||
        (!handRef.current && !poseRef.current && !faceRef.current) ||
        video.readyState < 2
      )
        return;
      const nowMs = performance.now();
      const results: {
        leftHand: RawLandmark[] | null;
        rightHand: RawLandmark[] | null;
        pose: RawLandmark[] | null;
        face: RawLandmark[] | null;
      } = { leftHand: null, rightHand: null, pose: null, face: null };

      try {
        if (handRef.current) {
          const hr = handRef.current.detectForVideo(video, nowMs) as HandLandmarkerResult;
          const left: RawLandmark[] = [];
          const right: RawLandmark[] = [];
          for (let i = 0; i < hr.landmarks.length; i++) {
            const pts: RawLandmark[] = hr.landmarks[i].map((lm) => ({
              x: lm.x, y: lm.y, z: lm.z,
            }));
            const h = hr.handednesses[i]?.[0]?.categoryName;
            if (h === "Left") left.push(...pts);
            else right.push(...pts);
          }
          results.leftHand = left.length ? left : null;
          results.rightHand = right.length ? right : null;
        }
        if (poseRef.current) {
          const pr = poseRef.current.detectForVideo(video, nowMs) as PoseLandmarkerResult;
          if (pr.landmarks[0]) {
            results.pose = pr.landmarks[0].map((lm) => ({ x: lm.x, y: lm.y, z: lm.z }));
          }
        }
        if (faceRef.current) {
          const fr = faceRef.current.detectForVideo(video, nowMs) as FaceLandmarkerResult;
          if (fr.landmarks[0]) {
            results.face = fr.landmarks[0].map((lm) => ({ x: lm.x, y: lm.y, z: lm.z }));
          }
        }
      } catch {
        // detection may fail on some frames
      }

      // Draw overlay
      const canvas = frameRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const draw = (pts: RawLandmark[] | null, color: string) => {
          if (!pts) return;
          ctx.fillStyle = color;
          for (const lm of pts) {
            ctx.beginPath();
            ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
            ctx.fill();
          }
        };
        draw(results.leftHand, "#06b6d4");
        draw(results.rightHand, "#3b82f6");
        draw(results.pose, "#10b981");
        draw(results.face, "#a855f7");
      }

      // Stream features to backend via WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const payload = {
          left_hand: results.leftHand,
          right_hand: results.rightHand,
          pose: results.pose,
          face: results.face,
        };
        try {
          wsRef.current.send(
            JSON.stringify({ session_id: sessionId, frames: [payload] })
          );
        } catch {
          // ignore feature errors
        }
      }
    },
    [ready, sessionId]
  );

  // Start/stop camera
  const toggle = useCallback(async () => {
    if (isActive) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
      setIsActive(false);
      onStatus?.("Recognition camera stopped.");
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
      setIsActive(true);
      setError(null);
      onStatus?.("Recognition camera active.");
      lastTimeRef.current = performance.now();
      rafRef.current = requestAnimationFrame(function loop(time: number) {
        if (!isActive) return;
        const d = time - lastTimeRef.current;
        lastTimeRef.current = time;
        countRef.current += 1;
        if (d >= 1000) {
          setFps(countRef.current);
          countRef.current = 0;
          lastTimeRef.current = time;
        }
        detect(videoRef.current!);
        rafRef.current = requestAnimationFrame(loop);
      });
    } catch {
      setError("Camera access denied or unavailable.");
      onStatus?.("Camera error — recognition unavailable.");
    }
  }, [isActive, detect, onStatus]);

  return {
    isActive,
    ready,
    error,
    fps,
    videoRef,
    frameRef,
    toggle,
  };
}
