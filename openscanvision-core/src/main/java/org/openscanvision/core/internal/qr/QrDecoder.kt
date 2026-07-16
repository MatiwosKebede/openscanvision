package org.openscanvision.core.internal.qr

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.suspendCancellableCoroutine
import org.openscanvision.core.internal.omr.CardDetector
import org.openscanvision.core.internal.omr.Templates
import kotlin.coroutines.resume

internal object QrDecoder {
    private const val TAG = "QrDecoder"

    private val scanner by lazy {
        BarcodeScanning.getClient(
            BarcodeScannerOptions.Builder()
                .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                .build()
        )
    }

    suspend fun decode(bitmap: Bitmap): String? {
        val image = InputImage.fromBitmap(bitmap, 0)
        return try {
            suspendCancellableCoroutine { cont ->
                scanner.process(image)
                    .addOnSuccessListener { barcodes ->
                        val text = barcodes.firstOrNull { it.format == Barcode.FORMAT_QR_CODE }?.rawValue
                            ?: barcodes.firstOrNull()?.rawValue
                        if (cont.isActive) cont.resume(text)
                    }
                    .addOnFailureListener { e ->
                        Log.w(TAG, "QR decode failed", e)
                        if (cont.isActive) cont.resume(null)
                    }
            }
        } catch (e: Exception) {
            Log.w(TAG, "QR decode error", e)
            null
        }
    }

    suspend fun decodeFromOriginalFrame(
        frameBitmap: Bitmap,
        cardCorners: List<PointF>,
        standardizedFallback: Bitmap? = null
    ): Pair<String, List<PointF>>? {
        if (cardCorners.size == 4) {
            val templateCardCorners = listOf(
                PointF(0f, 0f),
                PointF((Templates.REF_WIDTH - 1).toFloat(), 0f),
                PointF((Templates.REF_WIDTH - 1).toFloat(), (Templates.REF_HEIGHT - 1).toFloat()),
                PointF(0f, (Templates.REF_HEIGHT - 1).toFloat())
            )
            val homography = CardDetector.HomographySolver.solve(templateCardCorners, cardCorners)
            if (homography != null) {
                val qrCornersInFrame = CardDetector.predictImagePoints(Templates.SHARED_QR_CORNERS, homography)
                val result = decodeCroppedRegion(frameBitmap, qrCornersInFrame, paddingFraction = 0.5f)
                if (result != null) return Pair(result, qrCornersInFrame)
            }
        }

        decode(frameBitmap)?.let { return Pair(it, emptyList()) }
        return standardizedFallback?.let { decodeQrRegion(it)?.let { text -> Pair(text, emptyList()) } }
    }

    private suspend fun decodeCroppedRegion(
        bitmap: Bitmap,
        corners: List<PointF>,
        paddingFraction: Float
    ): String? {
        if (corners.isEmpty()) return null

        val minX = corners.minOf { it.x }
        val minY = corners.minOf { it.y }
        val maxX = corners.maxOf { it.x }
        val maxY = corners.maxOf { it.y }

        val padX = ((maxX - minX) * paddingFraction).coerceAtLeast(4f)
        val padY = ((maxY - minY) * paddingFraction).coerceAtLeast(4f)

        val left = (minX - padX).toInt().coerceIn(0, bitmap.width - 1)
        val top = (minY - padY).toInt().coerceIn(0, bitmap.height - 1)
        val right = (maxX + padX).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = (maxY + padY).toInt().coerceIn(top + 1, bitmap.height)

        if (right <= left || bottom <= top) return null

        val crop = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        return try {
            decode(crop)
        } finally {
            crop.recycle()
        }
    }

    suspend fun decodeQrRegion(standardizedBitmap: Bitmap): String? {
        val corners = Templates.SHARED_QR_CORNERS
        if (corners.size != 4) return decode(standardizedBitmap)

        decodeCroppedRegion(standardizedBitmap, corners, paddingFraction = 0.25f)?.let { return it }
        return decode(standardizedBitmap)
    }
}
