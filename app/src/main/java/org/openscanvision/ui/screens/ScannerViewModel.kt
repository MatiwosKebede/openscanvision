package org.openscanvision.ui.screens

import android.graphics.Bitmap
import android.graphics.PointF
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

class ScannerViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(ScannerUiState())
    val uiState: StateFlow<ScannerUiState> = _uiState

    private val captureRequested = AtomicBoolean(false)
    private val trackingState = TrackingState()

    // Flag to track if QR has been successfully decoded for current scan
    private var qrDecoded = false

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
     * Manual capture bypasses the QR requirement.
     */
    fun requestCapture() {
        captureRequested.set(true)
        qrDecoded = true  // allow capture even if QR not found
        _uiState.update { it.copy(captureStatus = "Manual capture requested...") }
    }

    fun dismissResult() {
        // Reset QR flag after result is dismissed (so next scan re‑requires QR)
        qrDecoded = false
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

    fun processFrame(imageProxy: ImageProxy) {
        if (imageProxy.image == null) {
            imageProxy.close()
            return
        }

        // ─── Extract bitmap and gray mat early ──────────────────────────────
        val gray = yPlaneToGrayMat(imageProxy)
        val frameBitmap = imageProxy.toBitmap()
        if (frameBitmap == null) {
            gray.release()
            imageProxy.close()
            return
        }

        // ─── Update tracking ──────────────────────────────────────────────────
        val trackingResult = updateTracking(
            gray = gray,
            trackingState = trackingState,
            captureRequested = { captureRequested.get() }
        )

        // Update UI with tracking info
        _uiState.update {
            it.copy(
                isTracking = trackingResult.isTracking,
                captureStatus = if (trackingResult.isStable) "Card detected – stable" else it.captureStatus
            )
        }

        // ─── Auto‑capture logic (strict: 4 markers + QR) ────────────────────
        if (trackingResult.isStable && !captureRequested.get()) {
            // If QR already decoded, auto‑capture immediately
            if (qrDecoded) {
                captureRequested.set(true)
                _uiState.update { it.copy(captureStatus = "Auto‑capturing...") }
            } else {
                // Otherwise, try to decode QR from this frame (only once)
                _uiState.update { it.copy(captureStatus = "QR not detected – waiting...") }
                // Launch QR decoding in background using the extracted bitmap
                viewModelScope.launch {
                    val corners = cardCornersFromArUco(trackingResult.arUcoMap)
                        ?: CardDetector.detectCardCorners(frameBitmap)
                    if (corners != null && corners.size == 4) {
                        val qrResult = QrDecoder.decodeFromOriginalFrame(
                            frameBitmap = frameBitmap,       // <-- use the extracted bitmap
                            cardCorners = corners,
                            standardizedFallback = null
                        )
                        if (qrResult != null) {
                            qrDecoded = true
                            _uiState.update { it.copy(qrToken = qrResult.first) }
                            // Now trigger auto‑capture
                            captureRequested.set(true)
                            _uiState.update { it.copy(captureStatus = "QR found – auto‑capturing...") }
                        } else {
                            _uiState.update { it.copy(captureStatus = "QR not found – hold steady") }
                        }
                    }
                }
            }
        }

        // ─── Perform capture if requested ────────────────────────────────────
        if (captureRequested.compareAndSet(true, false)) {
            performCapture(frameBitmap, trackingResult.arUcoMap)
        }

        // ─── Clean up ─────────────────────────────────────────────────────────
        gray.release()
        imageProxy.close()
    }

    private fun performCapture(frameBitmap: Bitmap, arUcoMap: Map<Int, List<PointF>>) {
        val cardCorners = cardCornersFromArUco(arUcoMap) ?: CardDetector.detectCardCorners(frameBitmap)
        if (cardCorners == null || cardCorners.size != 4) {
            _uiState.update { it.copy(captureStatus = "Card shape estimation failed. Retrying...") }
            return
        }

        viewModelScope.launch {
            val warped = OpenCVUtils.warpCard(frameBitmap, cardCorners, Templates.REF_WIDTH, Templates.REF_HEIGHT)
            if (warped == null) {
                _uiState.update { it.copy(captureStatus = "Warping failed. Retrying...") }
                return@launch
            }

            val cleaned = ImagePreprocessor.enhanceContrast(warped)
            val standardized = ImagePreprocessor.denoise(cleaned)

            val template = Templates.CANDIDATE // or detect from QR

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
            val decodedQrText = qrResult?.first ?: _uiState.value.qrToken

            val annotatedBitmap = OMRExtractor.renderAnnotatedImage(standardized, template, report, decodedQrText)
            val filledOneBased = report.allFilled.map { it + 1 }

            trackingState.metricsSuccesses++
            if (trackingState.lockStartFrame != -1) {
                trackingState.metricsTotalLockFrames += (trackingState.frameCounter - trackingState.lockStartFrame)
                trackingState.metricsTotalTimeToCaptureMs += (System.currentTimeMillis() - trackingState.lockStartTimeMs)
            }

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