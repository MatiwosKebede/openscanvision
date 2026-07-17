package org.openscanvision.ui.screens

import android.graphics.Bitmap
import android.graphics.PointF
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.openscanvision.core.internal.omr.CardDetector
import org.openscanvision.core.internal.omr.ImagePreprocessor
import org.openscanvision.core.internal.omr.OMRExtractor
import org.openscanvision.core.internal.omr.OpenCVUtils
import org.openscanvision.core.internal.omr.Templates
import org.openscanvision.core.internal.qr.QrDecoder
import org.openscanvision.utils.toBitmap
import java.util.concurrent.atomic.AtomicBoolean
import org.openscanvision.ui.screens.polygonArea
import org.openscanvision.ui.screens.validateCaptureQuality

class ScannerViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(ScannerUiState())
    val uiState: StateFlow<ScannerUiState> = _uiState

    private val captureRequested = AtomicBoolean(false)
    private val trackingState = TrackingState()

    // QR state for the current scan
    private var qrDecoded = false
    private var qrText = ""
    private var isScanning = false  // prevents overlapping scans

    private var previewView: PreviewView? = null
    private var cameraError: String? = null

    fun setPreviewView(view: PreviewView) {
        previewView = view
    }

    fun setCameraError(message: String) {
        cameraError = message
        _uiState.update { it.copy(captureStatus = "Camera error: $message") }
    }

    /**
     * Manual capture – still requires QR (no bypass).
     */
    fun requestCapture() {
        if (!qrDecoded) {
            _uiState.update { it.copy(captureStatus = "Please wait – QR not yet detected") }
            return
        }
        captureRequested.set(true)
        _uiState.update { it.copy(captureStatus = "Manual capture requested...") }
    }

    fun dismissResult() {
        // Reset everything for the next scan
        qrDecoded = false
        qrText = ""
        isScanning = false
        captureRequested.set(false)
        _uiState.update {
            it.copy(
                scanResultBitmap = null,
                filledIndices = emptyList(),
                confidence = 0f,
                templateName = null,
                qrToken = null,
                voterId = "",
                voterName = ""
            )
        }
    }

    @OptIn(ExperimentalGetImage::class)  // <-- FIX: suppresses Lint error for experimental API
    fun processFrame(imageProxy: ImageProxy) {
        if (imageProxy.image == null) {
            imageProxy.close()
            return
        }

        val gray = yPlaneToGrayMat(imageProxy)
        val frameBitmap = imageProxy.toBitmap()
        if (frameBitmap == null) {
            gray.release()
            imageProxy.close()
            return
        }

        val trackingResult = updateTracking(
            gray = gray,
            trackingState = trackingState,
            captureRequested = { captureRequested.get() }
        )

        // Decrement cooldowns
        if (trackingState.autoCaptureCooldown > 0) trackingState.autoCaptureCooldown--
        if (trackingState.quickRetryDelay > 0) trackingState.quickRetryDelay--

        // Update UI status
        val status = when {
            isScanning -> "Processing scan..."
            trackingResult.isStable && trackingState.stableFrameCounter >= AUTO_CAPTURE_STABLE_FRAMES -> {
                if (qrDecoded) "QR detected – ready to capture"
                else "Card stable – waiting for QR..."
            }
            trackingResult.isStable -> "Card detected – stabilizing..."
            else -> "Position card in frame"
        }
        _uiState.update {
            it.copy(
                isTracking = trackingResult.isTracking,
                captureStatus = status,
                qrToken = if (qrDecoded) qrText else it.qrToken
            )
        }

        // ─── Auto‑capture logic ──────────────────────────────────────────────
        val shouldAutoCapture = !isScanning &&
                trackingResult.isStable &&
                trackingState.stableFrameCounter >= AUTO_CAPTURE_STABLE_FRAMES &&
                trackingState.autoCaptureCooldown == 0 &&
                trackingState.quickRetryDelay == 0 &&
                !captureRequested.get()

        if (shouldAutoCapture) {
            // If QR already decoded, capture immediately
            if (qrDecoded) {
                captureRequested.set(true)
                _uiState.update { it.copy(captureStatus = "Auto‑capturing...") }
            } else {
                // Try to decode QR once
                _uiState.update { it.copy(captureStatus = "Scanning for QR...") }
                viewModelScope.launch {
                    val cardCorners = cardCornersFromArUco(trackingResult.arUcoMap)
                    if (cardCorners != null && cardCorners.size == 4) {
                        val qrResult = QrDecoder.decodeFromOriginalFrame(
                            frameBitmap = frameBitmap,
                            cardCorners = cardCorners,
                            standardizedFallback = null
                        )
                        if (qrResult != null) {
                            val prefix = qrResult.first.take(3)
                            if (prefix == "VX" || prefix == "AGN") {
                                qrDecoded = true
                                qrText = qrResult.first
                                _uiState.update { it.copy(qrToken = qrResult.first) }
                                // Now trigger capture
                                captureRequested.set(true)
                                _uiState.update { it.copy(captureStatus = "QR found – auto‑capturing...") }
                            } else {
                                _uiState.update { it.copy(captureStatus = "Invalid QR prefix – waiting for correct card") }
                            }
                        } else {
                            _uiState.update { it.copy(captureStatus = "QR not found – hold steady") }
                        }
                    } else {
                        _uiState.update { it.copy(captureStatus = "No ArUco corners for QR – holding") }
                    }
                }
            }
        }

        // ─── Perform capture if requested ────────────────────────────────────
        if (captureRequested.compareAndSet(true, false)) {
            performCapture(frameBitmap, trackingResult.arUcoMap)
        }

        gray.release()
        imageProxy.close()
    }

    private fun performCapture(frameBitmap: Bitmap, arUcoMap: Map<Int, List<PointF>>) {
        // Prevent overlapping scans
        if (isScanning) return
        isScanning = true

        // ─── Step 1: Validate marker quality ──────────────────────────────────
        val (valid, reason) = validateCaptureQuality(arUcoMap, trackingState.markerConfidence)
        if (!valid) {
            trackingState.quickRetryDelay = QUICK_RETRY_DELAY_FRAMES
            _uiState.update { it.copy(captureStatus = "Quality reject: $reason. Retrying...") }
            isScanning = false
            return
        }

        // ─── Step 2: Get card corners ──────────────────────────────────────
        val cardCorners = cardCornersFromArUco(arUcoMap)
        if (cardCorners == null || cardCorners.size != 4) {
            trackingState.quickRetryDelay = QUICK_RETRY_DELAY_FRAMES
            _uiState.update { it.copy(captureStatus = "Not enough ArUco markers. Retrying...") }
            isScanning = false
            return
        }

        // ─── Step 3: Validate quadrilateral area ────────────────────────────
        val area = polygonArea(cardCorners)
        if (area < 100f) {
            trackingState.quickRetryDelay = QUICK_RETRY_DELAY_FRAMES
            _uiState.update { it.copy(captureStatus = "Invalid corners. Retrying...") }
            isScanning = false
            return
        }

        // ─── Step 4: Verify QR is decoded ──────────────────────────────────
        if (!qrDecoded) {
            // Fallback: try to decode QR one more time
            viewModelScope.launch {
                val qrResult = QrDecoder.decodeFromOriginalFrame(
                    frameBitmap = frameBitmap,
                    cardCorners = cardCorners,
                    standardizedFallback = null
                )
                if (qrResult != null && (qrResult.first.take(3) == "VX" || qrResult.first.take(3) == "AGN")) {
                    qrDecoded = true
                    qrText = qrResult.first
                    _uiState.update { it.copy(qrToken = qrResult.first) }
                    // Proceed with capture
                    proceedWithCapture(frameBitmap, cardCorners)
                } else {
                    trackingState.quickRetryDelay = QUICK_RETRY_DELAY_FRAMES
                    _uiState.update { it.copy(captureStatus = "QR missing – cannot capture. Retrying...") }
                    isScanning = false
                }
            }
            return
        }

        // ─── Proceed with capture ──────────────────────────────────────────────
        proceedWithCapture(frameBitmap, cardCorners)
    }

    private fun proceedWithCapture(frameBitmap: Bitmap, cardCorners: List<PointF>) {
        viewModelScope.launch {
            val warped = OpenCVUtils.warpCard(frameBitmap, cardCorners, Templates.REF_WIDTH, Templates.REF_HEIGHT)
            if (warped == null) {
                trackingState.quickRetryDelay = QUICK_RETRY_DELAY_FRAMES
                _uiState.update { it.copy(captureStatus = "Warping failed. Retrying...") }
                isScanning = false
                return@launch
            }

            val cleaned = ImagePreprocessor.enhanceContrast(warped)
            val standardized = ImagePreprocessor.denoise(cleaned)

            val template = Templates.CANDIDATE // or from QR

            val omrDeferred = async(Dispatchers.Default) {
                OMRExtractor.readBubbleGroupsDetailed(standardized, template)
            }
            val qrDeferred = async(Dispatchers.Default) {
                QrDecoder.decodeFromOriginalFrame(
                    frameBitmap = frameBitmap,
                    cardCorners = cardCorners,
                    standardizedFallback = standardized
                )
            }

            val report = omrDeferred.await()
            val qrResult = qrDeferred.await()
            val decodedQrText = qrResult?.first ?: qrText

            val annotatedBitmap = OMRExtractor.renderAnnotatedImage(standardized, template, report, decodedQrText)
            val filledOneBased = report.allFilled.map { it + 1 }

            trackingState.metricsSuccesses++
            if (trackingState.lockStartFrame != -1) {
                trackingState.metricsTotalLockFrames += (trackingState.frameCounter - trackingState.lockStartFrame)
                trackingState.metricsTotalTimeToCaptureMs += (System.currentTimeMillis() - trackingState.lockStartTimeMs)
            }

            trackingState.autoCaptureCooldown = AUTO_CAPTURE_COOLDOWN_FRAMES
            isScanning = false

            val rejectRate = if (trackingState.metricsAttempts > 0) {
                (trackingState.metricsQualityRejects * 100f) / trackingState.metricsAttempts
            } else 0f

            val avgLock = if (trackingState.metricsSuccesses > 0) {
                trackingState.metricsTotalLockFrames.toFloat() / trackingState.metricsSuccesses
            } else 0f
            val avgTime = if (trackingState.metricsSuccesses > 0) {
                trackingState.metricsTotalTimeToCaptureMs / trackingState.metricsSuccesses
            } else 0L

            val metricsLine = "Attempts: ${trackingState.metricsAttempts} | Success: ${trackingState.metricsSuccesses} | " +
                    "Avg lock: ${"%.1f".format(avgLock)}f | Avg time: ${avgTime}ms | Reject: ${"%.1f".format(rejectRate)}%"

            withContext(Dispatchers.Main) {
                _uiState.update {
                    it.copy(
                        scanResultBitmap = annotatedBitmap,
                        filledIndices = filledOneBased,
                        confidence = report.overallConfidence,
                        templateName = template.name,
                        qrToken = decodedQrText,
                        voterId = "VOTER001",
                        voterName = "Abebech Demissie",
                        captureStatus = if (decodedQrText != null) {
                            "Success! ${report.allFilled.size} marked, QR read."
                        } else {
                            "Success! ${report.allFilled.size} marked. QR not read."
                        },
                        metricsLine = metricsLine
                    )
                }
            }
        }
    }
}