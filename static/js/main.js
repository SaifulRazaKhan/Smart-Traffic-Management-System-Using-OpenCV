/**
 * Smart Traffic Monitoring System - Frontend Controller
 * Academic Minor Project - B.Tech Computer Science
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentUploadedFile = null;
    let currentMediaType = "video"; // "video" or "image"
    let currentJobId = null;
    let pollInterval = null;

    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("video-file-input");
    const btnStartAnalysis = document.getElementById("btn-start-analysis");
    const btnReset = document.getElementById("btn-reset");
    const previewPlayer = document.getElementById("preview-player");
    const previewImage = document.getElementById("preview-image");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const uploadErrorBox = document.getElementById("upload-error-box");
    const sampleVideosContainer = document.getElementById("sample-videos-container");
    const sampleImagesContainer = document.getElementById("sample-images-container");

    // Metadata Elements
    const metaFilename = document.getElementById("meta-filename");
    const metaType = document.getElementById("meta-type");
    const metaResolution = document.getElementById("meta-resolution");
    const metaFps = document.getElementById("meta-fps");
    const metaDuration = document.getElementById("meta-duration");
    const metaFrames = document.getElementById("meta-frames");

    // Progress Elements
    const progressCard = document.getElementById("analysis-progress-card");
    const progressStatusTitle = document.getElementById("progress-status-title");
    const progressSubtext = document.getElementById("progress-subtext");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressPercentageLabel = document.getElementById("progress-percentage-label");
    const progressFrameCounter = document.getElementById("progress-frame-counter");
    const progressFpsCounter = document.getElementById("progress-fps-counter");
    const progressTimeCounter = document.getElementById("progress-time-counter");
    const analysisErrorBox = document.getElementById("analysis-error-box");
    const resultsDisplayArea = document.getElementById("results-display-area");

    // Dashboard Cards
    const cardTotalVehicles = document.getElementById("card-total-vehicles");
    const cardActiveVehicles = document.getElementById("card-active-vehicles");
    const cardCars = document.getElementById("card-cars");
    const cardMotorcycles = document.getElementById("card-motorcycles");
    const cardBuses = document.getElementById("card-buses");
    const cardTrucks = document.getElementById("card-trucks");
    const cardDensityBadge = document.getElementById("card-density-badge");
    const cardDensityExplanation = document.getElementById("card-density-explanation");
    const quickDensityBadge = document.getElementById("quick-density-badge");

    // Results View Elements
    const densityBanner = document.getElementById("density-banner");
    const bannerDensityTitle = document.getElementById("banner-density-title");
    const bannerDensityDescription = document.getElementById("banner-density-description");
    const bannerDensityCode = document.getElementById("banner-density-code");
    const processedCardTitle = document.getElementById("processed-card-title");
    const processedPlayer = document.getElementById("processed-player");
    const processedImage = document.getElementById("processed-image");
    const btnDownloadVideo = document.getElementById("btn-download-video");

    // Result Metric Table Elements
    const resTotalVehicles = document.getElementById("res-total-vehicles");
    const resCars = document.getElementById("res-cars");
    const resMotorcycles = document.getElementById("res-motorcycles");
    const resBuses = document.getElementById("res-buses");
    const resTrucks = document.getElementById("res-trucks");
    const resAvgActive = document.getElementById("res-avg-active");
    const resPeak = document.getElementById("res-peak");
    const resDensityBadge = document.getElementById("res-density-badge");
    const resModel = document.getElementById("res-model");
    const resFrames = document.getElementById("res-frames");
    const resTime = document.getElementById("res-time");
    const resFps = document.getElementById("res-fps");

    // Threshold inputs
    const inputLow = document.getElementById("input-low-threshold");
    const inputMed = document.getElementById("input-med-threshold");

    // Initialize
    loadSampleMedia();
    setupDropZone();
    setupNavigation();

    // -------------------------------------------------------------
    // File Upload & Drag-and-Drop
    // -------------------------------------------------------------
    function setupDropZone() {
        dropZone.addEventListener("click", () => fileInput.click());

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                handleFileUpload(fileInput.files[0]);
            }
        });

        btnReset.addEventListener("click", resetAll);
        btnStartAnalysis.addEventListener("click", startAnalysis);
    }

    async function handleFileUpload(file) {
        uploadErrorBox.style.display = "none";
        uploadErrorBox.textContent = "";

        const formData = new FormData();
        formData.append("video", file);

        dropZone.style.opacity = "0.6";
        dropZone.style.pointerEvents = "none";

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (!data.success) {
                showUploadError(data.error || "Failed to upload media file.");
                return;
            }

            const mediaType = data.file_type || (file.type.startsWith("image/") ? "image" : "video");
            setMediaSelected(data.filename, data.preview_url, data.metadata || {}, mediaType);
        } catch (err) {
            showUploadError("Network or server error while uploading file: " + err.message);
        } finally {
            dropZone.style.opacity = "1";
            dropZone.style.pointerEvents = "auto";
        }
    }

    function showUploadError(msg) {
        uploadErrorBox.textContent = msg;
        uploadErrorBox.style.display = "block";
    }

    function setMediaSelected(filename, previewUrl, metadata, mediaType) {
        currentUploadedFile = filename;
        currentMediaType = mediaType || (metadata.file_type || "video");

        previewPlaceholder.style.display = "none";

        if (currentMediaType === "image") {
            if (previewPlayer) {
                previewPlayer.pause();
                previewPlayer.style.display = "none";
            }
            if (previewImage) {
                previewImage.style.display = "block";
                previewImage.src = previewUrl;
            }
            if (metaType) metaType.textContent = "Traffic Photograph (Image)";
            metaFps.textContent = "N/A (Single Frame)";
            metaDuration.textContent = "Static Frame";
            metaFrames.textContent = "1 frame";
        } else {
            if (previewImage) {
                previewImage.style.display = "none";
            }
            if (previewPlayer) {
                previewPlayer.style.display = "block";
                previewPlayer.src = previewUrl;
                previewPlayer.load();
            }
            if (metaType) metaType.textContent = "Surveillance Video Stream";
            metaFps.textContent = metadata.fps ? `${metadata.fps} FPS` : "-";
            metaDuration.textContent = metadata.duration_seconds ? `${metadata.duration_seconds}s` : "-";
            metaFrames.textContent = metadata.total_frames || "-";
        }

        // Update common metadata
        metaFilename.textContent = filename;
        metaResolution.textContent = metadata.width && metadata.height ? `${metadata.width} x ${metadata.height}` : "-";

        // Enable analysis button
        btnStartAnalysis.disabled = false;

        // Clear active class on sample buttons
        document.querySelectorAll(".btn-sample").forEach(btn => btn.classList.remove("active-sample"));
    }

    // -------------------------------------------------------------
    // Bundled Sample Media (Videos & Images)
    // -------------------------------------------------------------
    async function loadSampleMedia() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();

            // Populate sample videos
            if (data.success && data.samples && data.samples.length > 0) {
                sampleVideosContainer.innerHTML = "";
                data.samples.forEach(sample => {
                    const btn = document.createElement("button");
                    btn.className = "btn-sample";
                    btn.textContent = sample.title;
                    btn.addEventListener("click", () => selectSampleMedia(sample.filename, btn, "video"));
                    sampleVideosContainer.appendChild(btn);
                });
            } else if (sampleVideosContainer) {
                sampleVideosContainer.innerHTML = "<small class='text-muted'>No sample videos found.</small>";
            }

            // Populate sample images
            if (data.success && data.sample_images && data.sample_images.length > 0) {
                sampleImagesContainer.innerHTML = "";
                data.sample_images.forEach(sample => {
                    const btn = document.createElement("button");
                    btn.className = "btn-sample";
                    btn.textContent = sample.title;
                    btn.addEventListener("click", () => selectSampleMedia(sample.filename, btn, "image"));
                    sampleImagesContainer.appendChild(btn);
                });
            } else if (sampleImagesContainer) {
                sampleImagesContainer.innerHTML = "<small class='text-muted'>No sample images found.</small>";
            }

        } catch (e) {
            if (sampleVideosContainer) sampleVideosContainer.innerHTML = "<small class='text-muted'>Samples offline.</small>";
            if (sampleImagesContainer) sampleImagesContainer.innerHTML = "<small class='text-muted'>Samples offline.</small>";
        }
    }

    async function selectSampleMedia(filename, buttonEl, mediaType) {
        uploadErrorBox.style.display = "none";
        try {
            const res = await fetch("/api/use-sample", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: filename })
            });
            const data = await res.json();
            if (data.success) {
                document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active-sample"));
                if (buttonEl) buttonEl.classList.add("active-sample");
                const mType = data.file_type || mediaType || "video";
                setMediaSelected(data.filename, data.preview_url, data.metadata || {}, mType);
            } else {
                showUploadError(data.error || "Could not load sample media.");
            }
        } catch (e) {
            showUploadError("Error selecting sample: " + e.message);
        }
    }

    // -------------------------------------------------------------
    // Traffic Analysis Execution & Polling
    // -------------------------------------------------------------
    async function startAnalysis() {
        if (!currentUploadedFile) {
            showUploadError("Please upload or select a traffic video or image first.");
            return;
        }

        const lowThresh = parseInt(inputLow.value) || 2;
        const medThresh = parseInt(inputMed.value) || 5;

        btnStartAnalysis.disabled = true;
        btnReset.disabled = true;
        analysisErrorBox.style.display = "none";
        resultsDisplayArea.style.display = "none";

        // Show progress card
        progressCard.style.display = "block";
        progressBarFill.style.width = "0%";
        progressPercentageLabel.textContent = "0%";

        if (currentMediaType === "image") {
            progressStatusTitle.textContent = "Analyzing Traffic Image...";
            progressSubtext.textContent = "Running single-frame YOLO vehicle detection and ROI density classification...";
            progressFrameCounter.textContent = "1 / 1 Frame";
            progressFpsCounter.textContent = "Single inference";
            progressTimeCounter.textContent = "Processing...";
            progressBarFill.style.width = "50%";
            progressPercentageLabel.textContent = "50%";
        } else {
            progressStatusTitle.textContent = "Processing Traffic Video...";
            progressSubtext.textContent = "Executing YOLO vehicle detection and temporal tracking...";
            progressFrameCounter.textContent = "Initializing...";
            progressFpsCounter.textContent = "-";
            progressTimeCounter.textContent = "0.0s";
        }

        // Smooth scroll to results section
        document.getElementById("results").scrollIntoView({ behavior: "smooth" });

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: currentUploadedFile,
                    low_threshold: lowThresh,
                    medium_threshold: medThresh
                })
            });

            const data = await response.json();
            if (!data.success) {
                showAnalysisError(data.error || "Failed to initiate analysis.");
                progressCard.style.display = "none";
                btnStartAnalysis.disabled = false;
                btnReset.disabled = false;
                return;
            }

            // Image analysis finishes synchronously in backend!
            if (data.file_type === "image" || data.status === "completed") {
                progressBarFill.style.width = "100%";
                progressPercentageLabel.textContent = "100%";
                setTimeout(() => {
                    progressCard.style.display = "none";
                    resetButtons();
                    displayFinalResults(data.results, data.output_url, "image");
                }, 300);
                return;
            }

            // Video analysis runs asynchronously in background thread
            currentJobId = data.job_id;
            pollJobStatus(currentJobId);
        } catch (err) {
            showAnalysisError("Server communication error: " + err.message);
            progressCard.style.display = "none";
            btnStartAnalysis.disabled = false;
            btnReset.disabled = false;
        }
    }

    function pollJobStatus(jobId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${jobId}`);
                const data = await res.json();

                if (!data.success || !data.job) {
                    clearInterval(pollInterval);
                    showAnalysisError("Unable to retrieve job progress.");
                    resetButtons();
                    return;
                }

                const job = data.job;

                // Update Progress UI
                const pct = job.progress || 0;
                progressBarFill.style.width = `${pct}%`;
                progressPercentageLabel.textContent = `${pct}%`;

                if (job.frame && job.total_frames) {
                    progressFrameCounter.textContent = `${job.frame} / ${job.total_frames}`;
                }
                if (job.current_fps) {
                    progressFpsCounter.textContent = `${job.current_fps} FPS`;
                }
                if (job.elapsed_seconds) {
                    progressTimeCounter.textContent = `${job.elapsed_seconds}s`;
                }

                // Live updates to dashboard active vehicles and density
                if (job.active_vehicles !== undefined && cardActiveVehicles) {
                    cardActiveVehicles.textContent = job.active_vehicles;
                }
                if (job.current_density) {
                    updateDensityBadge(job.current_density);
                }
                if (job.running_counts) {
                    updateDashboardCards(job.running_counts, job.current_density);
                }

                // Status Completed
                if (job.status === "completed") {
                    clearInterval(pollInterval);
                    progressCard.style.display = "none";
                    resetButtons();
                    displayFinalResults(job.results, job.output_url, "video");
                } else if (job.status === "error") {
                    clearInterval(pollInterval);
                    progressCard.style.display = "none";
                    resetButtons();
                    showAnalysisError(job.error_message || "Video processing encountered an error.");
                }

            } catch (e) {
                console.error("Polling error:", e);
            }
        }, 800);
    }

    function resetButtons() {
        btnStartAnalysis.disabled = false;
        btnReset.disabled = false;
    }

    function showAnalysisError(msg) {
        analysisErrorBox.textContent = msg;
        analysisErrorBox.style.display = "block";
    }

    // -------------------------------------------------------------
    // Display Final Results & Update Dashboard
    // -------------------------------------------------------------
    function displayFinalResults(results, outputUrl, mediaType) {
        if (!results) return;

        const isImage = (mediaType === "image") || (results.media_type === "image");
        const counts = results.cumulative_counts || results.counts || {};
        const activeInfo = results.active_vehicles || {};
        const density = results.density || {};
        const perf = results.performance || {};

        // 1. Update Dashboard Metric Cards
        updateDashboardCards(counts, density);
        if (cardActiveVehicles) {
            if (isImage) {
                cardActiveVehicles.textContent = activeInfo.peak !== undefined ? activeInfo.peak : (counts.total_vehicles || 0);
            } else if (activeInfo.average !== undefined) {
                cardActiveVehicles.textContent = `${activeInfo.average} (avg)`;
            }
        }

        // 2. Update Density Banner
        densityBanner.className = `density-banner banner-${density.code ? density.code.toLowerCase() : 'low'}`;
        bannerDensityTitle.textContent = density.level || "Low Traffic";
        bannerDensityDescription.textContent = density.description || "";
        bannerDensityCode.textContent = density.code || "LOW";

        // 3. Update Output Media Viewer
        if (isImage) {
            if (processedPlayer) {
                processedPlayer.pause();
                processedPlayer.style.display = "none";
            }
            if (processedImage) {
                processedImage.style.display = "block";
                processedImage.src = outputUrl;
            }
            if (processedCardTitle) processedCardTitle.textContent = "Processed Image with Detections";
            if (btnDownloadVideo) {
                btnDownloadVideo.href = outputUrl;
                btnDownloadVideo.download = "annotated_traffic.jpg";
                btnDownloadVideo.innerHTML = "&#128190; Download Image";
            }
        } else {
            if (processedImage) {
                processedImage.style.display = "none";
            }
            if (processedPlayer) {
                processedPlayer.style.display = "block";
                processedPlayer.src = outputUrl;
                processedPlayer.load();
            }
            if (processedCardTitle) processedCardTitle.textContent = "Processed Video with Bounding Boxes";
            if (btnDownloadVideo) {
                btnDownloadVideo.href = outputUrl;
                btnDownloadVideo.download = "annotated_traffic.mp4";
                btnDownloadVideo.innerHTML = "&#128190; Download Video";
            }
        }

        // 4. Update Metrics Table
        resTotalVehicles.textContent = counts.total_vehicles || 0;
        resCars.textContent = counts.cars || 0;
        resMotorcycles.textContent = counts.motorcycles || 0;
        resBuses.textContent = counts.buses || 0;
        resTrucks.textContent = counts.trucks || 0;

        if (resAvgActive) {
            if (isImage) {
                resAvgActive.textContent = `${activeInfo.peak || counts.total_vehicles || 0} in ROI`;
            } else {
                resAvgActive.textContent = activeInfo.average !== undefined ? `${activeInfo.average} veh/frame` : "-";
            }
        }
        resPeak.textContent = activeInfo.peak !== undefined ? activeInfo.peak : (counts.peak_simultaneous_vehicles || 0);

        resDensityBadge.textContent = density.level || "Low Traffic";
        resDensityBadge.className = `badge-density ${density.badge_class || 'density-low'}`;

        resModel.textContent = results.model_used || "YOLO26n";
        resFrames.textContent = perf.total_frames_processed || (isImage ? "1 frame" : "-");
        resTime.textContent = perf.processing_time_seconds ? `${perf.processing_time_seconds}s` : "-";
        resFps.textContent = perf.processing_fps ? `${perf.processing_fps} FPS` : "-";

        // Reveal results area
        resultsDisplayArea.style.display = "block";
    }

    function updateDensityBadge(density) {
        if (!density) return;
        cardDensityBadge.textContent = density.code || "LOW";
        cardDensityBadge.className = `badge-density-large ${density.badge_class || 'density-low'}`;
        cardDensityExplanation.textContent = density.description || density.level || "Low Traffic";

        quickDensityBadge.textContent = density.level || "Low Traffic";
        quickDensityBadge.className = `badge-density ${density.badge_class || 'density-low'}`;
    }

    function updateDashboardCards(counts, density) {
        cardTotalVehicles.textContent = counts.total_vehicles ?? 0;
        cardCars.textContent = counts.cars ?? 0;
        cardMotorcycles.textContent = counts.motorcycles ?? 0;
        cardBuses.textContent = counts.buses ?? 0;
        cardTrucks.textContent = counts.trucks ?? 0;

        if (density) {
            updateDensityBadge(density);
        }
    }

    function resetAll() {
        if (pollInterval) clearInterval(pollInterval);
        currentUploadedFile = null;
        currentJobId = null;

        fileInput.value = "";
        if (previewPlayer) {
            previewPlayer.pause();
            previewPlayer.removeAttribute("src");
            previewPlayer.style.display = "none";
        }
        if (previewImage) {
            previewImage.removeAttribute("src");
            previewImage.style.display = "none";
        }
        if (processedPlayer) {
            processedPlayer.pause();
            processedPlayer.removeAttribute("src");
            processedPlayer.style.display = "none";
        }
        if (processedImage) {
            processedImage.removeAttribute("src");
            processedImage.style.display = "none";
        }

        previewPlaceholder.style.display = "block";

        metaFilename.textContent = "-";
        if (metaType) metaType.textContent = "-";
        metaResolution.textContent = "-";
        metaFps.textContent = "-";
        metaDuration.textContent = "-";
        metaFrames.textContent = "-";

        btnStartAnalysis.disabled = true;
        uploadErrorBox.style.display = "none";
        analysisErrorBox.style.display = "none";
        progressCard.style.display = "none";
        resultsDisplayArea.style.display = "none";

        updateDashboardCards({
            total_vehicles: 0,
            cars: 0,
            motorcycles: 0,
            buses: 0,
            trucks: 0
        }, null);
        if (cardActiveVehicles) cardActiveVehicles.textContent = "0";

        cardDensityBadge.textContent = "PENDING";
        cardDensityBadge.className = "badge-density-large density-neutral";
        cardDensityExplanation.textContent = "Run analysis to evaluate density";

        quickDensityBadge.textContent = "Awaiting Analysis";
        quickDensityBadge.className = "badge-density density-neutral";

        document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active-sample"));
    }

    // -------------------------------------------------------------
    // Navigation Scroll Spy
    // -------------------------------------------------------------
    function setupNavigation() {
        const sections = document.querySelectorAll("section");
        const navLinks = document.querySelectorAll(".nav-link");

        window.addEventListener("scroll", () => {
            let current = "";
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                if (window.pageYOffset >= sectionTop - 120) {
                    current = section.getAttribute("id");
                }
            });

            navLinks.forEach(link => {
                link.classList.remove("active");
                if (link.getAttribute("href") === `#${current}`) {
                    link.classList.add("active");
                }
            });
        });
    }
});
