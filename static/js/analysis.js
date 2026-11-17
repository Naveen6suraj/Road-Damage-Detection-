/**
 * ROADGUARD AI v3.0 — FIELD INTELLIGENCE
 * Module: static/js/analysis.js
 * Analysis Studio Workstation Frontend Controller
 */

let selectedImageFile = null;
let selectedVideoFile = null;
let activeVideoJobId = null;
let videoPollInterval = null;

let currentOriginalImageUrl = null;
let currentAnnotatedImageUrl = null;
let currentImageZoomScale = 1.0;

// --------------------------------------------------------------------------
// STUDIO TAB SWITCHING
// --------------------------------------------------------------------------
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
        if (!sections[key] || !buttons[key]) continue;
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
// CONFIDENCE SLIDER PERSISTENCE & CONTROL
// --------------------------------------------------------------------------
function updateConfidenceDisplay(val) {
    const display = document.getElementById("imgConfDisplay");
    if (display) display.innerText = val + "%";
    sessionStorage.setItem("roadguard_active_conf", val);
}

// --------------------------------------------------------------------------
// IMAGE SELECTION, METADATA & PREVIEW
// --------------------------------------------------------------------------
function handleImageSelection(file) {
    if (!file) return;

    // Validate MIME type
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
        alert("Invalid file format. Please upload a JPG, PNG, or WEBP pavement photo.");
        return;
    }

    // Validate size (max 50 MB)
    if (file.size > 50 * 1024 * 1024) {
        alert("File size exceeds 50 MB limit. Please select a smaller photo.");
        return;
    }

    selectedImageFile = file;
    currentOriginalImageUrl = URL.createObjectURL(file);
    currentAnnotatedImageUrl = null;
    currentImageZoomScale = 1.0;

    // Populate Metadata Card
    const metaCard = document.getElementById("imageFileMetaCard");
    const dropZone = document.getElementById("imageDropZone");
    const thumb = document.getElementById("imageFileThumb");
    const nameEl = document.getElementById("selectedFileName");
    const sizeEl = document.getElementById("selectedFileSize");
    const typeEl = document.getElementById("selectedFileType");
    const dimsEl = document.getElementById("selectedFileDims");

    if (nameEl) nameEl.innerText = file.name;
    if (sizeEl) sizeEl.innerText = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    if (typeEl) typeEl.innerText = (file.type ? file.type.replace("image/", "").toUpperCase() : "JPG");
    if (thumb) thumb.src = currentOriginalImageUrl;

    // Extract natural image dimensions
    const probeImg = new Image();
    probeImg.src = currentOriginalImageUrl;
    probeImg.onload = function() {
        if (dimsEl) dimsEl.innerText = `${probeImg.naturalWidth} × ${probeImg.naturalHeight} px`;
    };

    if (metaCard) metaCard.classList.remove("hidden");
    if (dropZone) dropZone.classList.add("hidden");

    // Render viewport preview (Fit-to-view, no distortion)
    renderViewportImage(currentOriginalImageUrl, "Roadway Image Preview (Ready for Inspection)");

    // Enable button only if model is ready
    const btn = document.getElementById("btnAnalyzeImage");
    if (btn) {
        if (typeof MODEL_INFERENCE_ENABLED !== "undefined" && !MODEL_INFERENCE_ENABLED) {
            btn.disabled = true;
            btn.title = "Custom road damage model weights (best.pt) required";
        } else {
            btn.disabled = false;
            btn.title = "Execute YOLO11 damage & severity estimation";
        }
    }

    // Reset results & category pills to '—' Not analyzed
    resetCategoryCounters();
    const sevSection = document.getElementById("imageSeveritySection");
    const compareToggle = document.getElementById("imageCompareToggle");
    const actions = document.getElementById("imageActions");
    if (sevSection) sevSection.classList.add("hidden");
    if (compareToggle) compareToggle.classList.add("hidden");
    if (actions) actions.classList.add("hidden");
}

function removeSelectedImage() {
    selectedImageFile = null;
    currentOriginalImageUrl = null;
    currentAnnotatedImageUrl = null;
    currentImageZoomScale = 1.0;

    const fileInput = document.getElementById("imageFileInput");
    if (fileInput) fileInput.value = "";

    const metaCard = document.getElementById("imageFileMetaCard");
    const dropZone = document.getElementById("imageDropZone");
    if (metaCard) metaCard.classList.add("hidden");
    if (dropZone) dropZone.classList.remove("hidden");

    // Reset viewport to empty placeholder
    const container = document.getElementById("imagePreviewContainer");
    if (container) {
        container.innerHTML = `
            <div class="viewport-placeholder">
                <div class="viewport-placeholder-icon">
                    <svg class="icon icon-lg" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                </div>
                <div>
                    <div style="font-weight: 700; color: #D2D7D0; margin-bottom: 0.25rem;">No Roadway Image Loaded</div>
                    <div style="font-size: 0.82rem; color: #777D75;">Select or drop a pavement photo from the left panel to begin inspection</div>
                </div>
            </div>
        `;
    }

    // Disable button
    const btn = document.getElementById("btnAnalyzeImage");
    if (btn) {
        btn.disabled = true;
        btn.title = "Select an image to inspect";
    }

    // Reset counters & hide severity sections
    resetCategoryCounters();
    const sevSection = document.getElementById("imageSeveritySection");
    const compareToggle = document.getElementById("imageCompareToggle");
    const actions = document.getElementById("imageActions");
    if (sevSection) sevSection.classList.add("hidden");
    if (compareToggle) compareToggle.classList.add("hidden");
    if (actions) actions.classList.add("hidden");
}

function renderViewportImage(src, altText) {
    const container = document.getElementById("imagePreviewContainer");
    if (!container) return;
    container.innerHTML = `
        <div class="viewport-media-wrapper" id="imageMediaWrapper">
            <img id="viewportImg" src="${src}" alt="${altText}" class="viewport-media" style="transform: scale(${currentImageZoomScale});">
        </div>
    `;
    updateZoomBadge();
}

// --------------------------------------------------------------------------
// ZOOM & FULLSCREEN CONTROLS
// --------------------------------------------------------------------------
function adjustImageZoom(delta) {
    currentImageZoomScale = Math.max(0.5, Math.min(3.0, currentImageZoomScale + delta));
    const img = document.getElementById("viewportImg");
    if (img) img.style.transform = `scale(${currentImageZoomScale})`;
    updateZoomBadge();
}

function resetImageZoom() {
    currentImageZoomScale = 1.0;
    const img = document.getElementById("viewportImg");
    if (img) img.style.transform = "scale(1)";
    updateZoomBadge();
}

function updateZoomBadge() {
    const badge = document.getElementById("imageZoomBadge");
    if (badge) badge.innerText = Math.round(currentImageZoomScale * 100) + "%";
}

function toggleViewportFullscreen() {
    const container = document.getElementById("imagePreviewContainer");
    if (!container) return;
    if (!document.fullscreenElement) {
        if (container.requestFullscreen) {
            container.requestFullscreen();
        } else if (container.webkitRequestFullscreen) {
            container.webkitRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
    }
}

// --------------------------------------------------------------------------
// COMPARISON TOGGLE: ORIGINAL vs ANNOTATED
// --------------------------------------------------------------------------
function setImageCompareMode(mode) {
    const btnAnnotated = document.getElementById("btnViewAnnotated");
    const btnOriginal = document.getElementById("btnViewOriginal");
    const img = document.getElementById("viewportImg");

    if (mode === "original") {
        if (btnOriginal) btnOriginal.classList.add("active");
        if (btnAnnotated) btnAnnotated.classList.remove("active");
        if (img && currentOriginalImageUrl) img.src = currentOriginalImageUrl;
    } else {
        if (btnAnnotated) btnAnnotated.classList.add("active");
        if (btnOriginal) btnOriginal.classList.remove("active");
        if (img && currentAnnotatedImageUrl) img.src = currentAnnotatedImageUrl;
    }
}

// --------------------------------------------------------------------------
// RESET CATEGORY COUNTERS
// --------------------------------------------------------------------------
function resetCategoryCounters() {
    const classes = ["D00", "D10", "D20", "D40", "OTHER"];
    classes.forEach(c => {
        const countEl = document.getElementById("imgCount" + c);
        const pill = document.getElementById("pill" + c);
        const sub = document.getElementById("sub" + c);
        if (countEl) countEl.innerText = "—";
        if (pill) pill.classList.add("not-analyzed");
        if (sub) sub.innerText = "Not analyzed";
    });
}

function updateCategoryCounters(counts) {
    const classes = [
        { code: "D00", name: "Longitudinal" },
        { code: "D10", name: "Transverse" },
        { code: "D20", name: "Alligator" },
        { code: "D40", name: "Pothole" },
        { code: "OTHER", name: "Surface" }
    ];

    classes.forEach(cls => {
        const c = cls.code;
        const countVal = (counts && counts[c] !== undefined) ? counts[c] : 0;
        const countEl = document.getElementById("imgCount" + c);
        const pill = document.getElementById("pill" + c);
        const sub = document.getElementById("sub" + c);

        if (countEl) countEl.innerText = countVal;
        if (pill) {
            pill.classList.remove("not-analyzed");
            if (countVal > 0) {
                pill.style.borderColor = "var(--border-strong)";
            }
        }
        if (sub) {
            sub.innerText = countVal === 1 ? "1 defect" : `${countVal} defects`;
        }
    });
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

// --------------------------------------------------------------------------
// EXECUTE IMAGE INFERENCE
// --------------------------------------------------------------------------
async function executeImageAnalysis() {
    if (!selectedImageFile) {
        alert("Please select or drop a road image first.");
        return;
    }

    const btn = document.getElementById("btnAnalyzeImage");
    const loader = document.getElementById("imageLoader");
    const slider = document.getElementById("imgConfSlider");
    const confVal = slider ? (parseFloat(slider.value) / 100.0).toFixed(2) : "0.25";

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
            if (response.status === 503) {
                alert("MODEL OFFLINE\n\nYour image is ready for inspection, but the custom road-damage model is not installed. Add a compatible models/best.pt checkpoint to enable real road-damage detection.");
            } else {
                alert(`Analysis Notice (${response.status}): ${data.error || "Failed to process image"}`);
            }
            btn.disabled = false;
            loader.classList.add("hidden");
            return;
        }

        // Save annotated URL and show annotated comparison mode
        currentAnnotatedImageUrl = data.result_image;
        if (data.source_image) currentOriginalImageUrl = data.source_image;

        renderViewportImage(currentAnnotatedImageUrl, "Annotated Road Damage Detection Result");

        // Enable Comparison Toggle & Zoom controls
        const compareToggle = document.getElementById("imageCompareToggle");
        const btnAnnotated = document.getElementById("btnViewAnnotated");
        const btnOriginal = document.getElementById("btnViewOriginal");
        if (compareToggle) compareToggle.classList.remove("hidden");
        if (btnAnnotated) btnAnnotated.classList.add("active");
        if (btnOriginal) btnOriginal.classList.remove("active");

        // Update category counts
        updateCategoryCounters(data.counts || {});

        // Populate Dedicated Severity Summary
        const sevSection = document.getElementById("imageSeveritySection");
        const scoreEl = document.getElementById("imgSeverityScore");
        const badgeEl = document.getElementById("imgSeverityBadge");
        const totalDefectsEl = document.getElementById("imgTotalDefects");
        const slaEl = document.getElementById("imgSeveritySla");
        const latencyEl = document.getElementById("imageInferenceLatency");

        const summary = data.summary || {};
        const maxScore = summary.max_severity_score !== undefined ? summary.max_severity_score : 0.0;
        const maxLevel = summary.max_severity_level || "LOW";
        const totalCount = summary.total_defects !== undefined ? summary.total_defects : (data.detections ? data.detections.length : 0);

        if (scoreEl) scoreEl.innerText = Number(maxScore).toFixed(1);
        if (badgeEl) badgeEl.innerHTML = getSeverityBadgeHtml(maxLevel, maxScore);
        if (totalDefectsEl) totalDefectsEl.innerText = `${totalCount} Distress Zone${totalCount === 1 ? '' : 's'}`;
        if (latencyEl) latencyEl.innerText = `Latency: ${(data.processing_time_sec || 0.0).toFixed(2)}s`;

        // Service Level Agreement target
        if (slaEl) {
            switch (maxLevel) {
                case "CRITICAL":
                    slaEl.innerText = "Urgent Repair ≤ 24 Hours";
                    slaEl.style.color = "var(--critical)";
                    break;
                case "HIGH":
                    slaEl.innerText = "Priority Maintenance ≤ 7 Days";
                    slaEl.style.color = "var(--high)";
                    break;
                case "MODERATE":
                    slaEl.innerText = "Routine Repair ≤ 30 Days";
                    slaEl.style.color = "var(--moderate)";
                    break;
                default:
                    slaEl.innerText = "Routine Pavement Monitoring";
                    slaEl.style.color = "var(--low)";
            }
        }

        // Render defect table
        const tbody = document.querySelector("#imageDefectsTable tbody");
        tbody.innerHTML = "";

        if (data.detections && data.detections.length > 0) {
            data.detections.forEach((d, idx) => {
                const tr = document.createElement("tr");
                const sevBadge = getSeverityBadgeHtml(d.severity_level, d.severity_score);
                const bboxStr = d.bbox ? `[${d.bbox.map(n => Math.round(n)).join(", ")}]` : "—";
                const rawScore = d.raw_severity !== undefined ? Number(d.raw_severity).toFixed(1) : (d.severity_score ? Number(d.severity_score).toFixed(1) : "—");
                const adjScore = d.adjusted_severity !== undefined ? Number(d.adjusted_severity).toFixed(1) : (d.severity_score ? Number(d.severity_score).toFixed(1) : "—");

                tr.innerHTML = `
                    <td class="font-mono">${idx + 1}</td>
                    <td><strong>${d.class_name}</strong> <span class="badge badge-dark" style="font-size: 0.65rem;">${d.class_code}</span></td>
                    <td class="font-mono">${Math.round(d.confidence * 100)}%</td>
                    <td class="font-mono">${rawScore}</td>
                    <td class="font-mono">${adjScore}</td>
                    <td>${sevBadge}</td>
                    <td class="font-mono" style="font-size: 0.75rem; color: var(--text-muted);">${bboxStr}</td>
                    <td style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">${d.reason}</td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="8" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No pavement distress detected above ${Math.round(confVal * 100)}% confidence threshold. Surface appears clear.</td>`;
            tbody.appendChild(tr);
        }

        if (sevSection) sevSection.classList.remove("hidden");

        // Setup PDF / CSV download links
        const actions = document.getElementById("imageActions");
        const btnPdf = document.getElementById("btnDownloadImagePdf");
        const btnCsv = document.getElementById("btnDownloadImageCsv");

        if (btnPdf) btnPdf.href = `/api/reports/download/${data.analysis_id}/pdf`;
        if (btnCsv) btnCsv.href = `/api/reports/download/${data.analysis_id}/csv`;
        if (actions) actions.classList.remove("hidden");

        if (typeof showToast === "function") {
            showToast(`Inspection ${data.analysis_id} complete (${totalCount} defects detected).`, "success");
        }

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
    if (container) {
        container.innerHTML = `<video id="activeVideoPlayer" src="${url}" controls style="max-height: 480px; border-radius: var(--radius-sm); width: 100%; object-fit: contain;"></video>`;
    }

    // Populate Video Metadata Card
    const metaCard = document.getElementById("videoFileMetaCard");
    const dropZone = document.getElementById("videoDropZone");
    const nameEl = document.getElementById("selectedVideoName");
    const sizeEl = document.getElementById("selectedVideoSize");
    const durationEl = document.getElementById("selectedVideoDuration");
    const dimsEl = document.getElementById("selectedVideoDims");

    if (nameEl) nameEl.innerText = file.name;
    if (sizeEl) sizeEl.innerText = (file.size / (1024 * 1024)).toFixed(2) + " MB";

    // Probe duration and resolution
    const tempVideo = document.createElement("video");
    tempVideo.preload = "metadata";
    tempVideo.src = url;
    tempVideo.onloadedmetadata = function() {
        if (durationEl) durationEl.innerText = tempVideo.duration.toFixed(1) + "s";
        if (dimsEl) dimsEl.innerText = `${tempVideo.videoWidth} × ${tempVideo.videoHeight} px`;
    };

    if (metaCard) metaCard.classList.remove("hidden");
    if (dropZone) dropZone.classList.add("hidden");

    // Enable button if model ready
    const btn = document.getElementById("btnAnalyzeVideo");
    if (btn) {
        if (typeof MODEL_INFERENCE_ENABLED !== "undefined" && !MODEL_INFERENCE_ENABLED) {
            btn.disabled = true;
            btn.title = "Custom road damage model weights (best.pt) required";
        } else {
            btn.disabled = false;
            btn.title = "Execute multi-frame video tracking & deduplication";
        }
    }
}

function removeSelectedVideo() {
    selectedVideoFile = null;
    const fileInput = document.getElementById("videoFileInput");
    if (fileInput) fileInput.value = "";

    const metaCard = document.getElementById("videoFileMetaCard");
    const dropZone = document.getElementById("videoDropZone");
    if (metaCard) metaCard.classList.add("hidden");
    if (dropZone) dropZone.classList.remove("hidden");

    const container = document.getElementById("videoPreviewContainer");
    if (container) {
        container.innerHTML = `
            <div class="viewport-placeholder">
                <div class="viewport-placeholder-icon">
                    <svg class="icon icon-lg" viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/></svg>
                </div>
                <div>
                    <div style="font-weight: 700; color: #D2D7D0; margin-bottom: 0.25rem;">No Dashcam Video Loaded</div>
                    <div style="font-size: 0.82rem; color: #777D75;">Select or drop an MP4 / AVI road video to begin processing</div>
                </div>
            </div>
        `;
    }

    const btn = document.getElementById("btnAnalyzeVideo");
    if (btn) btn.disabled = true;

    const statsBox = document.getElementById("videoTrackingStatsBox");
    const actions = document.getElementById("videoActions");
    if (statsBox) statsBox.classList.add("hidden");
    if (actions) actions.classList.add("hidden");
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
    const slider = document.getElementById("vidConfSlider");
    const confVal = slider ? (parseFloat(slider.value) / 100.0).toFixed(2) : "0.25";

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
            if (response.status === 503) {
                alert("MODEL OFFLINE\n\nCustom road damage model weights (models/best.pt) required to process dashcam videos.");
            } else {
                alert(`Video Analysis Notice (${response.status}): ${data.error || "Failed to initiate video processing"}`);
            }
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
                progressFps.innerText = `${(job.fps_processing || 0.0).toFixed(1)} FPS`;

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
    if (container) {
        container.innerHTML = `<video src="${result.result_video}" controls autoplay style="max-height: 480px; border-radius: var(--radius-sm); width: 100%; object-fit: contain;"></video>`;
    }

    // Download buttons
    const actions = document.getElementById("videoActions");
    const btnVideo = document.getElementById("btnDownloadAnnotatedVideo");
    const btnPdf = document.getElementById("btnDownloadVideoPdf");
    const btnCsv = document.getElementById("btnDownloadVideoCsv");

    if (btnVideo) btnVideo.href = result.result_video;
    if (btnPdf) btnPdf.href = `/api/reports/download/${result.analysis_id}/pdf`;
    if (btnCsv) btnCsv.href = `/api/reports/download/${result.analysis_id}/csv`;
    if (actions) actions.classList.remove("hidden");

    // Populate Tracking Stats
    const statsBox = document.getElementById("videoTrackingStatsBox");
    const summary = result.summary || {};
    const uniqueCount = summary.unique_defects || (result.unique_defects ? result.unique_defects.length : 0);
    const totalFrames = summary.total_defects || 0;
    const dedupeRatio = totalFrames > 0 ? (totalFrames / Math.max(1, uniqueCount)).toFixed(1) + "x" : "1.0x";

    const uniqueEl = document.getElementById("vidStatUnique");
    const peakSevEl = document.getElementById("vidStatPeakSev");
    const peakScoreEl = document.getElementById("vidStatPeakScore");
    const frameDetsEl = document.getElementById("vidStatFrameDets");
    const dedupeRatioEl = document.getElementById("vidStatDedupeRatio");

    if (uniqueEl) uniqueEl.innerText = uniqueCount;
    if (peakSevEl) peakSevEl.innerText = summary.max_severity_level || "LOW";
    if (peakScoreEl) peakScoreEl.innerText = `${(summary.max_severity_score || 0.0).toFixed(1)} / 100`;
    if (frameDetsEl) frameDetsEl.innerText = totalFrames;
    if (dedupeRatioEl) dedupeRatioEl.innerText = dedupeRatio;

    // Populate Unique Defects Table
    const tbody = document.querySelector("#videoDefectsTable tbody");
    if (tbody) {
        tbody.innerHTML = "";
        const defects = result.unique_defects || [];
        if (defects.length > 0) {
            defects.forEach((d) => {
                const tr = document.createElement("tr");
                const sevBadge = getSeverityBadgeHtml(d.severity_level, d.severity_score);
                tr.innerHTML = `
                    <td class="font-mono"><strong>#${d.track_id}</strong></td>
                    <td><strong>${d.class_name}</strong> <span class="badge badge-dark" style="font-size: 0.65rem;">${d.class_code}</span></td>
                    <td class="font-mono">${Math.round(d.confidence * 100)}%</td>
                    <td>${sevBadge}</td>
                    <td class="font-mono" style="font-size: 0.8rem;">${d.hits || 1} frames (${(d.first_timestamp_sec || 0).toFixed(1)}s &ndash; ${(d.last_timestamp_sec || 0).toFixed(1)}s)</td>
                    <td style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">${d.reason}</td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No unique road distress tracks confirmed across video frames.</td>`;
            tbody.appendChild(tr);
        }
    }

    if (statsBox) statsBox.classList.remove("hidden");

    if (typeof showToast === "function") {
        showToast(`Video analysis complete (${uniqueCount} unique road defects tracked).`, "success");
    }
}

// --------------------------------------------------------------------------
// SEVERITY BADGE HTML HELPER
// --------------------------------------------------------------------------
function getSeverityBadgeHtml(level, score) {
    const s = score !== undefined && score !== null ? ` [${Number(score).toFixed(0)}]` : "";
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

// --------------------------------------------------------------------------
// CAMERA DEVICE ENUMERATION HELPER
// --------------------------------------------------------------------------
async function populateCameraDevices() {
    const select = document.getElementById("cameraDeviceSelect");
    if (!select || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        if (videoDevices.length > 0) {
            select.innerHTML = "";
            videoDevices.forEach((dev, idx) => {
                const opt = document.createElement("option");
                opt.value = dev.deviceId;
                opt.text = dev.label || `Camera ${idx + 1} (${dev.deviceId.slice(0, 8)}...)`;
                select.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn("Could not enumerate media devices:", e);
    }
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    // Restore saved confidence slider
    const savedConf = sessionStorage.getItem("roadguard_active_conf") || localStorage.getItem("roadguard_default_conf");
    if (savedConf) {
        const slider = document.getElementById("imgConfSlider");
        const display = document.getElementById("imgConfDisplay");
        if (slider && display) {
            slider.value = savedConf;
            display.innerText = savedConf + "%";
        }
    }

    // Populate camera devices if supported
    populateCameraDevices();
});
