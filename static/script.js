/**
 * ROADGUARD AI - Legacy Compatibility Script
 * File: static/script.js
 * Fixed all legacy path, typo, and event issues.
 */

let useWebcam = false;
let outputMediaPath = "";

const uploadTab = document.getElementById("uploadTab");
const liveTab = document.getElementById("liveTab");
const uploadSection = document.getElementById("uploadSection");
const liveSection = document.getElementById("liveSection");
const dropZone = document.getElementById("dropZone");
const mediaUpload = document.getElementById("mediaUpload");
const preview = document.getElementById("preview");
const webcam = document.getElementById("webcam");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");

if (uploadTab && liveTab) {
    uploadTab.onclick = () => {
        useWebcam = false;
        if (uploadSection) uploadSection.classList.remove("hidden");
        if (liveSection) liveSection.classList.add("hidden");
        uploadTab.classList.add("active");
        liveTab.classList.remove("active");
    };

    liveTab.onclick = () => {
        useWebcam = true;
        if (uploadSection) uploadSection.classList.add("hidden");
        if (liveSection) liveSection.classList.remove("hidden");
        liveTab.classList.add("active");
        uploadTab.classList.remove("active");
        startWebcam();
    };
}

if (dropZone && mediaUpload) {
    dropZone.onclick = () => mediaUpload.click();
    mediaUpload.onchange = () => {
        handleMedia(mediaUpload.files[0]);
    };
}

function handleMedia(file) {
    if (!file || !dropZone) return;
    const url = URL.createObjectURL(file);
    dropZone.innerHTML = "";
    if (file.type.startsWith("image")) {
        dropZone.innerHTML = `<img src="${url}" style="max-height: 350px;">`;
    } else {
        dropZone.innerHTML = `<video src="${url}" controls style="max-height: 350px;"></video>`;
    }
}

async function startWebcam() {
    if (!webcam) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
    } catch (e) {
        alert("Camera error: " + e.message);
    }
}

if (predictBtn) {
    predictBtn.onclick = async () => {
        if (loader) loader.classList.remove("hidden");
        const formData = new FormData();

        if (!useWebcam) {
            const file = mediaUpload ? mediaUpload.files[0] : null;
            if (!file) {
                alert("Upload file first");
                if (loader) loader.classList.add("hidden");
                return;
            }

            formData.append("file", file);

            if (file.type.startsWith("image")) {
                const res = await fetch("/predict_image", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                updateCounters(data.counts || {});
                showOutput(data.result_image);
            } else {
                const res = await fetch("/predict_video", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                updateCounters(data.counts || {});
                // Fixed previous typo result_videoz -> result_video
                showOutput(data.result_video || data.result_videoz);
            }
        } else {
            alert("Please use the Analysis Studio Live Camera tab for real-time inference.");
        }

        if (loader) loader.classList.add("hidden");
    };
}

function updateCounters(c) {
    const d00 = document.getElementById("D00");
    const d10 = document.getElementById("D10");
    const d20 = document.getElementById("D20");
    const d40 = document.getElementById("D40");
    const oth = document.getElementById("OTHER");

    if (d00) d00.innerText = c.D00 || 0;
    if (d10) d10.innerText = c.D10 || 0;
    if (d20) d20.innerText = c.D20 || 0;
    if (d40) d40.innerText = c.D40 || 0;
    if (oth) oth.innerText = c.OTHER || 0;
}

function showOutput(path) {
    if (!path || !dropZone) return;
    outputMediaPath = path;
    dropZone.innerHTML = "";
    if (path.endsWith(".mp4") || path.endsWith(".avi")) {
        dropZone.innerHTML = `<video src="${path}" controls autoplay style="max-height: 350px;"></video>`;
    } else {
        dropZone.innerHTML = `<img src="${path}" style="max-height: 350px;">`;
    }
}

const downloadBtn = document.getElementById("downloadBtn");
if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
        if (!outputMediaPath) {
            alert("No result to download");
            return;
        }
        const link = document.createElement("a");
        link.href = outputMediaPath;
        link.download = outputMediaPath.endsWith(".mp4") ? "annotated_result.mp4" : "annotated_result.jpg";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}
