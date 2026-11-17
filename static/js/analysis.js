/**
 * ROADGUARD AI - Analysis Studio Frontend Controller
 * File: static/js/analysis.js
 */

let selectedImageFile = null;
let selectedVideoFile = null;
let activeVideoJobId = null;
let videoPollInterval = null;

// Tab Switching
function switchStudioTab(tabName) {
    const sections = {
        image: document.getElementById("sectionImage"),
        video: document.getElementById("sectionVideo"),
        camera: document.getElementById("sectionCamera")
    };
    const buttons = {
        image: document.getElementById("tabBtnImage"),
        video: document.getElementById("tabBtnVideo"),
        camera: document.getElementById("tabBtnCamera")
    };

    for (const key in sections) {
        if (key === tabName) {
            sections[key].classList.remove("hidden");
            buttons[key].classList.add("active");
        } else {
            sections[key].classList.add("hidden");
            buttons[key].classList.remove("active");
        }
    }

    // Stop webcam if leaving camera tab
    if (tabName !== "camera" && typeof stopWebcamStream === "function") {
        stopWebcamStream();
    }
}

// --------------------------------------------------------------------------
// IMAGE ANALYSIS LOGIC
// --------------------------------------------------------------------------
function handleImageSelection(file) {
    if (!file) return;
    selectedImageFile = file;

    const url = URL.createObjectURL(file);
    const container = document.getElementById("imagePreviewContainer");
    container.innerHTML = `<img src="${url}" alt="Preview" style="max-height: 440px; border-radius: var(--radius-sm);">`;

    // Update Drop Zone label
    const zoneTitle = document.querySelector("#imageDropZone .upload-title");
    if (zoneTitle) zoneTitle.innerText = `Selected: ${file.name}`;
}

// Drag & drop listeners for image
const imgDropZone = document.getElementById("imageDropZone");
if (imgDropZone) {
    imgDropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        imgDropZone.classList.add("dragover");
    });
    imgDropZone.addEventListener("dragleave", () => {
        imgDropZone.classList.remove("dragover");
    });
    imgDropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        imgDropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleImageSelection(e.dataTransfer.files[0]);
        }
    });
}

async function executeImageAnalysis() {
    if (!selectedImageFile) {
        alert("Please select or drop a road image first.");
        return;
    }

    const btn = document.getElementById("btnAnalyzeImage");
    const loader = document.getElementById("imageLoader");
    const confVal = (parseFloat(document.getElementById("imgConfSlider").value) / 100.0).toFixed(2);

    btn.disabled = true;
    loader.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", selectedImageFile);
    formData.append("confidence", confVal);

    try {
        const response = await fetch("/api/analyze/image", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!data.success) {
            const msg = data.message ? `\n\n${data.message}` : "";
            alert(`Analysis Notice (${response.status}): ${data.error || "Failed to process image"}${msg}`);
            btn.disabled = false;
            loader.classList.add("hidden");
            return;
        }

        // Render annotated output image
        const container = document.getElementById("imagePreviewContainer");
        container.innerHTML = `<img src="${data.result_image}" alt="Annotated Detection Result" style="max-height: 440px; border-radius: var(--radius-sm);">`;

        // Update class count boxes
        const counts = data.counts || {};
        document.getElementById("imgCountD00").innerText = counts.D00 || 0;
        document.getElementById("imgCountD10").innerText = counts.D10 || 0;
        document.getElementById("imgCountD20").innerText = counts.D20 || 0;
        document.getElementById("imgCountD40").innerText = counts.D40 || 0;
        document.getElementById("imgCountOTHER").innerText = counts.OTHER || 0;

        // Render defect table
        const tableContainer = document.getElementById("imageDefectsTableContainer");
        const tbody = document.querySelector("#imageDefectsTable tbody");
        tbody.innerHTML = "";

        if (data.detections && data.detections.length > 0) {
            data.detections.forEach((d, idx) => {
                const tr = document.createElement("tr");
                const sevBadge = getSeverityBadgeHtml(d.severity_level, d.severity_score);
                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><strong>${d.class_name}</strong> (${d.class_code})</td>
                    <td>${Math.round(d.confidence * 100)}%</td>
                    <td>${sevBadge}</td>
                    <td style="font-size: 0.8rem; color: var(--text-secondary);">${d.reason}</td>
                `;
                tbody.appendChild(tr);
            });
            tableContainer.classList.remove("hidden");
        } else {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="5" style="text-align: center; color: var(--text-muted);">No pavement defects detected above ${Math.round(confVal * 100)}% confidence threshold.</td>`;
            tbody.appendChild(tr);
            tableContainer.classList.remove("hidden");
        }

        // Setup PDF / CSV download links
        const actions = document.getElementById("imageActions");
        const btnPdf = document.getElementById("btnDownloadImagePdf");
        const btnCsv = document.getElementById("btnDownloadImageCsv");

        btnPdf.href = `/api/reports/download/${data.analysis_id}/pdf`;
        btnCsv.href = `/api/reports/download/${data.analysis_id}/csv`;
        actions.classList.remove("hidden");

    } catch (err) {
        alert(`Server communication error: ${err.message}`);
    } finally {
        btn.disabled = false;
        loader.classList.add("hidden");
    }
}

// --------------------------------------------------------------------------
// DASHCAM VIDEO ANALYSIS LOGIC
// --------------------------------------------------------------------------
function handleVideoSelection(file) {
    if (!file) return;
    selectedVideoFile = file;

    const url = URL.createObjectURL(file);
    const container = document.getElementById("videoPreviewContainer");
    container.innerHTML = `<video src="${url}" controls style="max-height: 440px; border-radius: var(--radius-sm); width: 100%;"></video>`;

    const zoneTitle = document.querySelector("#videoDropZone .upload-title");
    if (zoneTitle) zoneTitle.innerText = `Selected: ${file.name}`;
}

const vidDropZone = document.getElementById("videoDropZone");
if (vidDropZone) {
    vidDropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        vidDropZone.classList.add("dragover");
    });
    vidDropZone.addEventListener("dragleave", () => {
        vidDropZone.classList.remove("dragover");
    });
    vidDropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        vidDropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleVideoSelection(e.dataTransfer.files[0]);
        }
    });
}

async function executeVideoAnalysis() {
    if (!selectedVideoFile) {
        alert("Please select or drop a dashcam video first.");
        return;
    }

    const btn = document.getElementById("btnAnalyzeVideo");
    const progressBox = document.getElementById("videoProgressContainer");
    const progressBar = document.getElementById("videoProgressBar");
    const progressPercent = document.getElementById("videoProgressPercent");
    const progressStats = document.getElementById("videoProgressStats");
    const progressFps = document.getElementById("videoProgressFps");
    const confVal = (parseFloat(document.getElementById("vidConfSlider").value) / 100.0).toFixed(2);

    btn.disabled = true;
    progressBox.classList.remove("hidden");
    progressBar.style.width = "0%";
    progressPercent.innerText = "0%";

    const formData = new FormData();
    formData.append("file", selectedVideoFile);
    formData.append("confidence", confVal);

    try {
        const response = await fetch("/api/analyze/video", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!data.success) {
            const msg = data.message ? `\n\n${data.message}` : "";
            alert(`Video Analysis Notice (${response.status}): ${data.error || "Failed to initiate video processing"}${msg}`);
            btn.disabled = false;
            progressBox.classList.add("hidden");
            return;
        }

        activeVideoJobId = data.job_id;

        // Poll progress
        videoPollInterval = setInterval(async () => {
            try {
                const pollRes = await fetch(`/api/video/progress/${activeVideoJobId}`);
                const pollData = await pollRes.json();
                if (!pollData.success) return;

                const job = pollData.job;
                progressBar.style.width = `${job.percent}%`;
                progressPercent.innerText = `${job.percent}%`;
                progressStats.innerText = `Frame ${job.current_frame || 0} / ${job.total_frames || 0} | Tracks: ${job.unique_defects_count || 0}`;
                progressFps.innerText = `${job.fps_processing || 0.0} FPS`;

                if (job.status === "completed") {
                    clearInterval(videoPollInterval);
                    btn.disabled = false;
                    renderVideoResults(job.result);
                } else if (job.status === "failed") {
                    clearInterval(videoPollInterval);
                    btn.disabled = false;
                    alert(`Video Processing Failed: ${job.error || "Unknown error"}`);
                }
            } catch (pollErr) {
                console.error("Polling error:", pollErr);
            }
        }, 800);

    } catch (err) {
        alert(`Server communication error: ${err.message}`);
        btn.disabled = false;
        progressBox.classList.add("hidden");
    }
}

function renderVideoResults(result) {
    if (!result) return;

    // Display annotated video player
    const container = document.getElementById("videoPreviewContainer");
    container.innerHTML = `<video src="${result.result_video}" controls autoplay style="max-height: 440px; border-radius: var(--radius-sm); width: 100%;"></video>`;

    // Download buttons
    const actions = document.getElementById("videoActions");
    const btnVideo = document.getElementById("btnDownloadAnnotatedVideo");
    const btnPdf = document.getElementById("btnDownloadVideoPdf");
    const btnCsv = document.getElementById("btnDownloadVideoCsv");

    btnVideo.href = result.result_video;
    btnPdf.href = `/api/reports/download/${result.analysis_id}/pdf`;
    btnCsv.href = `/api/reports/download/${result.analysis_id}/csv`;
    actions.classList.remove("hidden");

    // Populate Tracking Stats
    const statsBox = document.getElementById("videoTrackingStatsBox");
    const summary = result.summary || {};
    document.getElementById("vidStatUnique").innerText = summary.unique_defects || (result.unique_defects ? result.unique_defects.length : 0);
    document.getElementById("vidStatPeakSev").innerText = summary.max_severity_level || "LOW";
    document.getElementById("vidStatPeakScore").innerText = `Score: ${(summary.max_severity_score || 0.0).toFixed(1)}`;
    document.getElementById("vidStatFrameDets").innerText = summary.total_defects || 0;

    // Populate Unique Defects Table
    const tbody = document.querySelector("#videoDefectsTable tbody");
    tbody.innerHTML = "";

    const defects = result.unique_defects || [];
    if (defects.length > 0) {
        defects.forEach((d) => {
            const tr = document.createElement("tr");
            const sevBadge = getSeverityBadgeHtml(d.severity_level, d.severity_score);
            tr.innerHTML = `
                <td><strong>#${d.track_id}</strong></td>
                <td>${d.class_name} (${d.class_code})</td>
                <td>${Math.round(d.confidence * 100)}%</td>
                <td>${sevBadge}</td>
                <td>${d.hits || 1} frames (${(d.first_timestamp_sec || 0).toFixed(1)}s - ${(d.last_timestamp_sec || 0).toFixed(1)}s)</td>
                <td style="font-size: 0.8rem; color: var(--text-secondary);">${d.reason}</td>
            `;
            tbody.appendChild(tr);
        });
    } else {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="6" style="text-align: center; color: var(--text-muted);">No unique defects confirmed across video frames.</td>`;
        tbody.appendChild(tr);
    }

    statsBox.classList.remove("hidden");
}

function getSeverityBadgeHtml(level, score) {
    const s = score ? ` (${Number(score).toFixed(0)})` : "";
    switch (level) {
        case "CRITICAL":
            return `<span class="badge badge-critical">CRITICAL${s}</span>`;
        case "HIGH":
            return `<span class="badge badge-high">HIGH${s}</span>`;
        case "MODERATE":
            return `<span class="badge badge-moderate">MODERATE${s}</span>`;
        default:
            return `<span class="badge badge-low">LOW${s}</span>`;
    }
}
