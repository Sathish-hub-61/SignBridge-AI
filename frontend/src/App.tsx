import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { CollectPanel } from "./CollectPanel";
import { useSignRecognition } from "./useSignRecognition";

interface HealthData {
  status: string;
  version: string;
  environment: string;
  recognizer: string;
  llm_configured: boolean;
  primary_model: string;
}

interface TokenChip {
  id: string;
  token: string;
  confidence: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: string;
  tokens?: TokenChip[];
  quickReplies?: string[];
  timestamp: string;
}

const COMMON_SIGNS = [
  "HELP", "I", "NEED", "DOCTOR", "PAIN", "WHERE", "TOILET", "YES", "NO", "THANK_YOU"
];

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);
  const [mockMode, setMockMode] = useState<boolean>(true);
  const [typedInput, setTypedInput] = useState<string>("");
  const [tokens, setTokens] = useState<TokenChip[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [streamingReply, setStreamingReply] = useState<string>("");
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(false);
  const [showModal, setShowModal] = useState<boolean>(false);
  const [modalText, setModalText] = useState<string>("");
const [emergencyAlert, setEmergencyAlert] = useState<boolean>(false);
const [sessionId] = useState<string>(() => "session-" + Math.random().toString(36).substring(2, 9));
const [view, setView] = useState<"chat" | "collect">("chat");

// Sign recognition camera state (Phase 4)
const [recognitionStatus, setRecognitionStatus] = useState<string>("Idle");
const [recognitionError, setRecognitionError] = useState<string | null>(null);

const {
  isActive: recognitionActive,
  ready: recognitionReady,
  error: recognitionHookError,
  fps: recognitionFps,
  videoRef: recognitionVideoRef,
  frameRef: recognitionCanvasRef,
  toggle: toggleRecognition,
} = useSignRecognition({
  sessionId,
  onToken: (token) => {
    setTokens((prev) => [
      ...prev,
      {
        id: `recog-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        token: token.token,
        confidence: token.confidence,
      },
    ]);
    setRecognitionStatus(`Recognized: ${token.token} (${(token.confidence * 100).toFixed(0)}%)`);
  },
  onStatus: setRecognitionStatus,
});

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Poll backend health
  const checkHealth = async () => {
    setLoadingHealth(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/health");
      if (response.ok) {
        const data: HealthData = await response.json();
        setHealth(data);
      } else {
        setHealth(null);
      }
    } catch {
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Surface hook-level errors to local state
  useEffect(() => {
    setRecognitionError(recognitionHookError || null);
  }, [recognitionHookError]);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingReply]);

  // Speech synthesis helper
  const speakText = (text: string) => {
    if (!ttsEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  // Convert typed text into editable chips
  const handleAddTypedTokens = () => {
    if (!typedInput.trim()) return;
    const words = typedInput.trim().toUpperCase().split(/\s+/);
    const newChips: TokenChip[] = words.map((w, idx) => ({
      id: `${Date.now()}-${idx}-${w}`,
      token: w,
      confidence: 0.95, // Default confidence for manually typed glosses
    }));
    setTokens((prev) => [...prev, ...newChips]);
    setTypedInput("");
    inputRef.current?.focus();
  };

  const handleAddQuickSign = (sign: string) => {
    const newChip: TokenChip = {
      id: `${Date.now()}-${sign}`,
      token: sign,
      confidence: 0.95,
    };
    setTokens((prev) => [...prev, newChip]);
  };

  const handleRemoveToken = (id: string) => {
    setTokens((prev) => prev.filter((t) => t.id !== id));
  };

  const handleEditToken = (id: string, newToken: string) => {
    const clean = newToken.trim().toUpperCase();
    if (!clean) {
      handleRemoveToken(id);
      return;
    }
    setTokens((prev) =>
      prev.map((t) => (t.id === id ? { ...t, token: clean } : t))
    );
  };

  const handleConfidenceChange = (id: string, newConf: number) => {
    setTokens((prev) =>
      prev.map((t) => (t.id === id ? { ...t, confidence: Math.max(0.1, Math.min(1.0, newConf)) } : t))
    );
  };

  // Send tokens through Interpret + Respond pipeline
  const handleSendTokens = async () => {
    // If input has unsent text, add it to tokens first
    let currentTokens = [...tokens];
    if (typedInput.trim()) {
      const words = typedInput.trim().toUpperCase().split(/\s+/);
      const newChips: TokenChip[] = words.map((w, idx) => ({
        id: `${Date.now()}-${idx}-${w}`,
        token: w,
        confidence: 0.95,
      }));
      currentTokens = [...currentTokens, ...newChips];
      setTokens(currentTokens);
      setTypedInput("");
    }

    if (currentTokens.length === 0) return;

    setIsProcessing(true);
    setStreamingReply("");

    const payloadTokens = currentTokens.map((t) => ({
      token: t.token,
      confidence: t.confidence,
    }));

    try {
      // Use SSE streaming endpoint /chat/stream
      const response = await fetch("http://127.0.0.1:8000/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tokens: payloadTokens,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullReply = "";
      let quickReplies: string[] = [];

      if (reader) {
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const block of lines) {
            const trimmed = block.trim();
            if (!trimmed) continue;

            const eventMatch = trimmed.match(/^event:\s*(\w+)/);
            const dataMatch = trimmed.match(/data:\s*(.+)$/m);

            if (eventMatch && dataMatch) {
              const eventType = eventMatch[1];
              try {
                const data = JSON.parse(dataMatch[1]);

                if (eventType === "interpretation") {
                  if (data.intent === "emergency") {
                    setEmergencyAlert(true);
                  }
                  // Add user turn to chat with interpreted sentence and tokens
                  const userMsg: Message = {
                    id: "user-" + Date.now(),
                    role: "user",
                    content: data.sentence,
                    intent: data.intent,
                    tokens: [...currentTokens],
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                  };
                  setMessages((prev) => [...prev, userMsg]);
                  // Clear active token chips once submitted
                  setTokens([]);
                } else if (eventType === "chunk") {
                  fullReply += data.delta;
                  setStreamingReply((prev) => prev + data.delta);
                } else if (eventType === "done") {
                  if (data.full_reply) fullReply = data.full_reply;
                  if (data.quick_replies) quickReplies = data.quick_replies;
                }
              } catch (e) {
                console.error("Failed to parse SSE payload", e);
              }
            }
          }
        }
      }

      // Add assistant response to message history
      const finalReply = fullReply.trim() || "I understood your signs.";
      const assistantMsg: Message = {
        id: "assistant-" + Date.now(),
        role: "assistant",
        content: finalReply,
        quickReplies: quickReplies,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setStreamingReply("");
      speakText(finalReply);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to process signs";
      const errorMsg: Message = {
        id: "error-" + Date.now(),
        role: "assistant",
        content: `Sorry, an error occurred: ${msg}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
      setStreamingReply("");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleQuickReplyClick = (replyText: string) => {
    // When a quick reply is clicked, send it as typed input or direct prompt
    setTypedInput(replyText);
  };

  const handleNewConversation = async () => {
    try {
      await fetch(`http://127.0.0.1:8000/chat/reset?session_id=${sessionId}`, {
        method: "POST",
      });
    } catch {
      // Ignore network errors on reset
    }
    setMessages([]);
    setTokens([]);
    setTypedInput("");
    setStreamingReply("");
    setEmergencyAlert(false);
  };

  const openShowToOthers = (text: string) => {
    setModalText(text);
    setShowModal(true);
  };

  const isConnected = health?.status === "healthy";

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-logo" aria-hidden="true">🤟</div>
          <div>
            <h1 className="brand-title">SignBridge AI</h1>
            <p className="brand-subtitle">Context-Aware Conversational Indian Sign Language Assistant</p>
          </div>
        </div>

        {/* Global Controls & Status */}
        <div className="controls-group">
          {/* Mock Mode Toggle */}
          <label className="toggle-label" htmlFor="mock-mode-toggle" title="Toggle camera simulation mock mode">
            <input
              id="mock-mode-toggle"
              type="checkbox"
              checked={mockMode}
              onChange={(e) => setMockMode(e.target.checked)}
              aria-label="Toggle Mock Mode"
            />
            <span className="toggle-custom"></span>
            <span className="toggle-text">Mock Mode (Typed)</span>
          </label>

          {/* TTS Toggle */}
          <button
            id="tts-toggle-btn"
            className={`control-btn ${ttsEnabled ? "btn-active" : ""}`}
            onClick={() => setTtsEnabled(!ttsEnabled)}
            aria-label="Toggle Text to Speech"
            title="Read assistant replies aloud"
          >
            {ttsEnabled ? "🔊 Speech: ON" : "🔇 Speech: OFF"}
          </button>

          {/* New Conversation Button */}
          <button
            id="new-conv-btn"
            className="control-btn"
            onClick={handleNewConversation}
            title="Reset conversation memory"
            aria-label="Start new conversation"
          >
            ✨ New Chat
          </button>

          {/* Connection Status Pill */}
          <div
            id="backend-connection-badge"
            className={`status-pill ${isConnected ? "status-connected" : "status-disconnected"}`}
            aria-live="polite"
          >
            <span className="status-dot" aria-hidden="true"></span>
            <span className="status-text">
              {loadingHealth && !health
                ? "Backend: connecting..."
                : isConnected
                ? "Backend: connected"
                : "Backend: offline"}
            </span>
          </div>

          {/* View toggle: Chat <-> Collect */}
          <button
            id="view-toggle-btn"
            className="control-btn"
            onClick={() => setView(view === "chat" ? "collect" : "chat")}
            title={`Switch to ${view === "chat" ? "Data Collection" : "Chat"} view`}
          >
            {view === "chat" ? "📊 Collect Data" : "💬 Chat"}
          </button>
        </div>
      </header>

      {view === "chat" ? (
        <>
          {/* Emergency Alert Card */}
          {emergencyAlert && (
            <div className="emergency-card" role="alert">
              <div className="emergency-icon" aria-hidden="true">🚨</div>
              <div className="emergency-text">
                <strong>EMERGENCY DETECTED</strong>
                <p>
                  SignBridge interpreted an emergency intent. Contact emergency services immediately
                  at <strong>112</strong> (India). SignBridge AI is an assistive aid and does not
                  provide medical treatment.
                </p>
              </div>
              <div className="emergency-actions">
                <a
                  href="tel:112"
                  className="action-btn emergency-call"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Call emergency services 112"
                >
                  📞 Call 112
                </a>
                <button
                  className="action-btn emergency-dismiss"
                  onClick={() => setEmergencyAlert(false)}
                  aria-label="Dismiss emergency card"
                >
                  ✕ Dismiss
                </button>
              </div>
            </div>
          )}

      {/* Main Two-Column Layout */}
      <main className="chat-layout">
        {/* Left Column: Sign Input & Token Chips */}
        <section className="input-column">
          <div className="dashboard-card input-panel">
            <div className="panel-header">
              <h2 className="panel-title">Sign Input & Recognition</h2>
              <span className="mode-badge">{mockMode ? "Mock Mode Active" : "Camera Mode"}</span>
            </div>

            {/* Recognition camera controls (Phase 4) */}
            <div className="recognition-controls">
              <button
                id="recognition-camera-btn"
                className={`action-btn ${
                  recognitionActive ? "btn-stop" : recognitionReady ? "btn-start" : "btn-start"
                }`}
                onClick={toggleRecognition}
                disabled={!recognitionReady}
                title="Start camera-based sign recognition"
              >
                {recognitionActive ? "⏹ Stop Recognition" : recognitionReady ? "▶ Start Recognition Camera" : "Loading models…"}
              </button>
              <span className={`recognition-status ${recognitionError ? "status-error" : "status-ok"}`}>
                {recognitionError || recognitionStatus}
              </span>
              <span className={`recognition-status ${recognitionActive ? "status-ok" : "status-muted"}`}>
                {recognitionActive ? `FPS: ${recognitionFps}` : ""}
              </span>
            </div>

            {/* Camera preview (shown only when recognition is active) */}
            {recognitionActive && recognitionVideoRef && (
              <div className="recognition-preview">
                <video
                  ref={recognitionVideoRef}
                  autoPlay
                  muted
                  playsInline
                  className="recognition-video"
                />
                <canvas ref={recognitionCanvasRef} className="recognition-overlay" />
                <p className="recognition-hint">
                  Hold signs in frame. Recognized tokens appear below as editable chips — review,
                  edit, then tap <strong>Interpret & Send</strong>.
                </p>
              </div>
            )}

            {/* Quick Sign Shortcut Buttons */}
            <div className="quick-signs-row" aria-label="Common ISL Signs">
              <span className="quick-signs-label">Quick Signs:</span>
              <div className="quick-signs-chips">
                {COMMON_SIGNS.map((sign) => (
                  <button
                    key={sign}
                    className="quick-sign-btn"
                    onClick={() => handleAddQuickSign(sign)}
                    title={`Add ${sign} token`}
                  >
                    +{sign}
                  </button>
                ))}
              </div>
            </div>

            {/* Typed Gloss Input Bar */}
            <form
              className="typed-input-form"
              onSubmit={(e) => {
                e.preventDefault();
                handleAddTypedTokens();
              }}
            >
              <input
                ref={inputRef}
                id="typed-gloss-input"
                className="gloss-text-input"
                type="text"
                value={typedInput}
                onChange={(e) => setTypedInput(e.target.value)}
                placeholder="Type ISL glosses (e.g. HELP I NEED DOCTOR)..."
                aria-label="Type sign glosses"
              />
              <button
                id="add-token-btn"
                type="submit"
                className="action-btn secondary-btn"
                disabled={!typedInput.trim()}
              >
                Add Chips
              </button>
              <button
                id="send-tokens-btn"
                type="button"
                className="action-btn primary-btn"
                onClick={handleSendTokens}
                disabled={isProcessing || (tokens.length === 0 && !typedInput.trim())}
              >
                {isProcessing ? "Processing..." : "Interpret & Send ↵"}
              </button>
            </form>

            {/* Live Editable Token Chips Tray */}
            <div className="tokens-tray-container">
              <div className="tray-header">
                <span className="tray-title">Recognized Token Chips (Human-in-the-Loop)</span>
                {tokens.length > 0 && (
                  <button
                    className="clear-tokens-btn"
                    onClick={() => setTokens([])}
                    title="Clear token chips"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {tokens.length === 0 ? (
                <div className="empty-tokens-placeholder">
                  <span>No active sign tokens. Type glosses above or tap Quick Signs.</span>
                </div>
              ) : (
                <div className="token-chips-grid">
                  {tokens.map((t) => {
                    const isLowConf = t.confidence < 0.6;
                    return (
                      <div
                        key={t.id}
                        className={`token-chip ${isLowConf ? "chip-low-conf" : "chip-high-conf"}`}
                        title="Click text to edit, click ✕ to delete"
                      >
                        <input
                          className="chip-token-input"
                          value={t.token}
                          onChange={(e) => handleEditToken(t.id, e.target.value)}
                          aria-label={`Token text ${t.token}`}
                        />
                        <span
                          className="chip-conf-badge"
                          title="Confidence score. Click to toggle low/high for testing."
                          onClick={() => handleConfidenceChange(t.id, isLowConf ? 0.95 : 0.45)}
                        >
                          {(t.confidence * 100).toFixed(0)}%
                        </span>
                        <button
                          className="chip-delete-btn"
                          onClick={() => handleRemoveToken(t.id)}
                          aria-label={`Remove token ${t.token}`}
                        >
                          ✕
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Right Column: Chat Transcript & Responses */}
        <section className="chat-column">
          <div className="dashboard-card chat-panel">
            <div className="chat-header">
              <h2 className="panel-title">Conversation Transcript</h2>
              <div className="chat-actions">
                {messages.length > 0 && (
                  <button
                    className="show-others-btn"
                    onClick={() => {
                      const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
                      const lastUser = [...messages].reverse().find((m) => m.role === "user");
                      const text = lastAssistant?.content || lastUser?.content || "SignBridge AI";
                      openShowToOthers(text);
                    }}
                    title="Open large high-contrast display for nearby people"
                  >
                    👁️ Show to Others
                  </button>
                )}
              </div>
            </div>

            {/* Messages Feed */}
            <div className="messages-feed" id="messages-container" role="log" aria-live="polite">
              {messages.length === 0 && !streamingReply && (
                <div className="empty-chat-state">
                  <div className="empty-chat-icon">💬</div>
                  <h3>No Messages Yet</h3>
                  <p>
                    Use Mock Mode on the left to input sign glosses like <code>HELP I NEED</code> and tap <strong>Interpret & Send</strong> to test Stage A (Interpretation) and Stage B (Conversational Response).
                  </p>
                </div>
              )}

              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`chat-bubble-wrapper ${m.role === "user" ? "user-wrapper" : "assistant-wrapper"}`}
                >
                  <div className={`chat-bubble ${m.role === "user" ? "user-bubble" : "assistant-bubble"}`}>
                    <div className="bubble-meta">
                      <span className="bubble-speaker">
                        {m.role === "user" ? "Signer (ISL)" : "SignBridge Assistant"}
                      </span>
                      {m.intent && (
                        <span className={`intent-tag intent-${m.intent}`}>
                          {m.intent}
                        </span>
                      )}
                      <span className="bubble-time">{m.timestamp}</span>
                    </div>

                    {/* Source Token Chips for User Turns */}
                    {m.tokens && m.tokens.length > 0 && (
                      <div className="source-tokens-row">
                        <span className="source-tokens-label">Signs:</span>
                        {m.tokens.map((tk) => (
                          <span key={tk.id} className="history-token-tag">
                            {tk.token}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Sentence Content */}
                    <div className="bubble-content">{m.content}</div>

                    {/* Quick Replies for Assistant Turns */}
                    {m.quickReplies && m.quickReplies.length > 0 && (
                      <div className="quick-replies-tray">
                        {m.quickReplies.map((qr, idx) => (
                          <button
                            key={idx}
                            className="quick-reply-pill"
                            onClick={() => handleQuickReplyClick(qr)}
                          >
                            {qr}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Speak Button */}
                    <button
                      className="speak-bubble-btn"
                      onClick={() => speakText(m.content)}
                      title="Read aloud"
                      aria-label="Speak message aloud"
                    >
                      🔊
                    </button>
                  </div>
                </div>
              ))}

              {/* Streaming Assistant Response in Flight */}
              {isProcessing && (
                <div className="chat-bubble-wrapper assistant-wrapper">
                  <div className="chat-bubble assistant-bubble streaming-bubble">
                    <div className="bubble-meta">
                      <span className="bubble-speaker">SignBridge Assistant</span>
                      <span className="streaming-indicator">generating...</span>
                    </div>
                    <div className="bubble-content">
                      {streamingReply || <span className="typing-cursor">▋</span>}
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>
        </section>
      </main>

      {/* "Show to Others" Fullscreen High-Contrast Modal */}
      {showModal && (
        <div
          id="show-to-others-modal"
          className="fullscreen-modal"
          role="dialog"
          aria-modal="true"
          aria-labelledby="modal-title"
          onClick={() => setShowModal(false)}
        >
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-top">
              <h2 id="modal-title" className="modal-title">SignBridge — Display to Others</h2>
              <button
                className="modal-close-btn"
                onClick={() => setShowModal(false)}
                aria-label="Close fullscreen modal"
              >
                ✕ Close (Esc)
              </button>
            </div>
            <div className="modal-body-text">{modalText}</div>
            <div className="modal-bottom-hint">
              Tap anywhere outside or press ESC to return to chat
            </div>
          </div>
        </div>
      )}
        </>
      ) : (
        <CollectPanel />
      )}

      {/* Footer */}
      <footer className="app-footer">
        <p>SignBridge AI • Phase 1 Generative AI Layer • Privacy-First Architecture</p>
      </footer>
    </div>
  );
};

export default App;
