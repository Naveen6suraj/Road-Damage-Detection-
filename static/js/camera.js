/**
 * ROADGUARD AI - Live Camera Stream Controller
 * File: static/js/camera.js
 */

let cameraStream = null;
let isCameraRunning = false;
let cameraProcessingLoopId = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let calculatedFps = 0.0;

async function startWebcamStream() {
    const video = document.getElementById("liveWebcamVideo");
    const canvas = document.getElementById("liveWebcamCanvas");
    const dot = document.getElementById("cameraStatusDot");
    const statusText = document.getElementById("cameraStatusText");
    const btnStart = document.getElementById("btnStartCamera");
    const btnStop = document.getElementById("btnStopCamera");
    const statsBar = document.getElementById("cameraStatsBar");

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "environment" // Prefer road/rear camera on mobile
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        isCameraRunning = true;
        dot.style.background = "var(--sev-low)";
        dot.classList.add("online");
        statusText.innerText = "Camera Active (Inference Streaming)";

        btnStart.classList.add("hidden");
        btnStop.classList.remove("hidden");
        statsBar.classList.remove("hidden");

        // Swap to display canvas over video
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        canvas.classList.remove("hidden");
        video.classList.add("hidden");

        // Start frame capture loop
        lastFrameTime = performance.now();
        frameCount = 0;
        processCameraLoop();

    } catch (err) {
        console.error("Camera access error:", err);
        let msg = "Could not access camera.";
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            msg = "Camera permission was denied. Please allow camera permissions in your browser.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            msg = "No compatible camera hardware detected on this device.";
        }
        alert(msg);
    }
}

function stopWebcamStream() {
    if (!isCameraRunning && !cameraStream) return;

    isCameraRunning = false;
    if (cameraProcessingLoopId) {
        clearTimeout(cameraProcessingLoopId);
        cameraProcessingLoopId = null;
    }

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    const video = document.getElementById("liveWebcamVideo");
    const canvas = document.getElementById("liveWebcamCanvas");
    const dot = document.getElementById("cameraStatusDot");
    const statusText = document.getElementById("cameraStatusText");
    const btnStart = document.getElementById("btnStartCamera");
    const btnStop = document.getElementById("btnStopCamera");
    const statsBar = document.getElementById("cameraStatsBar");

    if (video) video.srcObject = null;
    if (canvas) canvas.classList.add("hidden");
    if (video) video.classList.remove("hidden");

    if (dot) {
        dot.style.background = "var(--text-muted)";
        dot.classList.remove("online");
    }
    if (statusText) statusText.innerText = "Camera Inactive";
    if (btnStart) btnStart.classList.remove("hidden");
    if (btnStop) btnStop.classList.add("hidden");
    if (statsBar) statsBar.classList.add("hidden");
}

async function processCameraLoop() {
    if (!isCameraRunning) return;

    const video = document.getElementById("liveWebcamVideo");
    const canvas = document.getElementById("liveWebcamCanvas");
    if (!video || !canvas || video.readyState < 2) {
        cameraProcessingLoopId = setTimeout(processCameraLoop, 100);
        return;
    }

    const ctx = canvas.getContext("2d");
    if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    // Capture current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.65);

    const tStart = performance.now();

    try {
        const response = await fetch("/api/camera/process_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image: dataUrl,
                confidence: 0.25
            })
        });

        const data = await response.json();
        const latency = Math.round(performance.now() - tStart);

        if (data.success && data.annotated_frame) {
            // Render annotated frame from backend
            const img = new Image();
            img.onload = () => {
                if (isCameraRunning) {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                }
            };
            img.src = data.annotated_frame;

            // Update stats
            document.getElementById("camActiveCount").innerText = data.defect_count || 0;
            const sevEl = document.getElementById("camHighestSev");
            sevEl.innerText = data.highest_severity_level || "LOW";
            sevEl.style.color = getSeverityColor(data.highest_severity_level);
            document.getElementById("camLatency").innerText = `${latency} ms`;
        }

        // Calculate streaming FPS
        frameCount++;
        const elapsed = (performance.now() - lastFrameTime) / 1000.0;
        if (elapsed >= 1.0) {
            calculatedFps = (frameCount / elapsed).toFixed(1);
            document.getElementById("camFps").innerText = `${calculatedFps} FPS`;
            frameCount = 0;
            lastFrameTime = performance.now();
        }

    } catch (e) {
        console.warn("Live camera frame processing error:", e);
    }

    if (isCameraRunning) {
        // Run next frame with slight pacing
        cameraProcessingLoopId = setTimeout(processCameraLoop, 150);
    }
}

function getSeverityColor(level) {
    switch (level) {
        case "CRITICAL": return "var(--sev-critical)";
        case "HIGH": return "var(--sev-high)";
        case "MODERATE": return "var(--sev-moderate)";
        default: return "var(--sev-low)";
    }
}
