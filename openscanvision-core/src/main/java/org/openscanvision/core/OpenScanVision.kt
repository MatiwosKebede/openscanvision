package org.openscanvision.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.openscanvision.core.internal.omr.CardDetector
import org.openscanvision.core.internal.omr.ImagePreprocessor
import org.openscanvision.core.internal.omr.OMRExtractor
import org.openscanvision.core.internal.omr.OpenCVUtils
import org.openscanvision.core.internal.omr.Templates
import org.openscanvision.core.internal.qr.QrDecoder
import org.opencv.android.OpenCVLoader

object OpenScanVision {
    private var isInitialized = false

    @JvmStatic
    fun initialize(context: Context): Boolean {
        return try {
            isInitialized = OpenCVLoader.initDebug()
            isInitialized
        } catch (e: Exception) {
            false
        }
    }

    @JvmStatic
    suspend fun scan(
        standardizedBitmap: Bitmap,
        options: ScanOptions = ScanOptions.Default,
        qrPayload: String? = null
    ): ScanResult {
        if (!isInitialized) return ScanResult.Error.OpenCVNotInitialized
        val start = System.currentTimeMillis()

        return withContext(Dispatchers.IO) {
            try {
                val template = if (qrPayload != null) {
                    Templates.fromPrefix(qrPayload.take(3)) ?: options.template
                } else {
                    options.template
                }

                var processed = standardizedBitmap
                if (options.enableClahe) processed = ImagePreprocessor.enhanceContrast(processed)
                processed = ImagePreprocessor.denoise(processed)

                val (indices, confidence) = OMRExtractor.readFilledBubbles(processed, template)

                if (confidence < options.confidenceThreshold) {
                    return@withContext ScanResult.Error.LowConfidence(confidence)
                }

                val json = buildJson(template.prefix, indices, confidence, qrPayload)

                val annotated = if (options.generateAnnotatedImage) {
                    val report = OMRExtractor.readBubbleGroupsDetailed(processed, template)
                    OMRExtractor.renderAnnotatedImage(processed, template, report, qrPayload)
                } else null

                ScanResult.Success(
                    token = qrPayload ?: template.prefix,
                    templateUsed = template.name,
                    filledIndices = indices,
                    confidence = confidence,
                    json = json,
                    latencyMs = System.currentTimeMillis() - start,
                    warpedBitmap = processed,
                    annotatedBitmap = annotated,
                    qrPayload = qrPayload
                )
            } catch (e: Exception) {
                ScanResult.Error.Generic(e.message ?: "Unknown error")
            }
        }
    }

    @JvmStatic
    suspend fun scanFromFrame(
        frameBitmap: Bitmap,
        options: ScanOptions = ScanOptions.Default,
        cardCorners: List<PointF>? = null   // optional – if null, library detects corners
    ): ScanResult {
        if (!isInitialized) return ScanResult.Error.OpenCVNotInitialized
        val start = System.currentTimeMillis()

        return withContext(Dispatchers.IO) {
            // Use provided corners, or fall back to edge detection
            val corners = cardCorners ?: CardDetector.detectCardCorners(frameBitmap)
            ?: return@withContext ScanResult.Error.NoCardDetected

            val targetW = (Templates.REF_WIDTH * options.warpScale).toInt()
            val targetH = (Templates.REF_HEIGHT * options.warpScale).toInt()
            val warped = OpenCVUtils.warpCard(frameBitmap, corners, targetW, targetH)
                ?: return@withContext ScanResult.Error.WarpFailed

            var qrPayload: String? = null
            var qrCorners: List<PointF>? = null
            if (options.enableQrDecoding) {
                val qrResult = QrDecoder.decodeFromOriginalFrame(
                    frameBitmap = frameBitmap,
                    cardCorners = corners,
                    standardizedFallback = warped
                )
                qrPayload = qrResult?.first
                qrCorners = qrResult?.second
            }

            val selectedTemplate = if (qrPayload != null) {
                Templates.fromPrefix(qrPayload.take(3)) ?: options.template
            } else {
                options.template
            }

            if (options.requireQrMatch && qrPayload == null) {
                return@withContext ScanResult.Error.TemplateMismatch(null)
            }
            if (options.requireQrMatch && qrPayload != null && Templates.fromPrefix(qrPayload.take(3)) == null) {
                return@withContext ScanResult.Error.TemplateMismatch(qrPayload.take(3))
            }

            var processed = warped
            if (options.enableClahe) processed = ImagePreprocessor.enhanceContrast(processed)
            processed = ImagePreprocessor.denoise(processed)

            val (indices, confidence) = OMRExtractor.readFilledBubbles(processed, selectedTemplate)

            if (confidence < options.confidenceThreshold) {
                return@withContext ScanResult.Error.LowConfidence(confidence)
            }

            val json = buildJson(selectedTemplate.prefix, indices, confidence, qrPayload)

            val annotated = if (options.generateAnnotatedImage) {
                val report = OMRExtractor.readBubbleGroupsDetailed(processed, selectedTemplate)
                OMRExtractor.renderAnnotatedImage(processed, selectedTemplate, report, qrPayload)
            } else null

            ScanResult.Success(
                token = qrPayload ?: selectedTemplate.prefix,
                templateUsed = selectedTemplate.name,
                filledIndices = indices,
                confidence = confidence,
                json = json,
                latencyMs = System.currentTimeMillis() - start,
                warpedBitmap = processed,
                annotatedBitmap = annotated,
                qrPayload = qrPayload,
                qrCorners = qrCorners
            )
        }
    }

    private fun buildJson(prefix: String, indices: List<Int>, confidence: Float, qr: String?): String {
        val map = mapOf(
            "template" to prefix,
            "filled" to indices,
            "confidence" to confidence,
            "qr" to (qr ?: "null")
        )
        return Gson().toJson(map)
    }
}