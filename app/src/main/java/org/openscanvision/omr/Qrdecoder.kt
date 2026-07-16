package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * QR decoding via ML Kit barcode-scanning (already a project dependency — no ZXing needed).
 * ML Kit's scanner is Task-based/async, so decode() is a suspend function; call it from a
 * coroutine (ScannerScreen already runs the capture pipeline inside coroutineScope.launch).
 */
object QrDecoder {
    private const val TAG = "QrDecoder"

    private val scanner by lazy {
        BarcodeScanning.getClient(
            BarcodeScannerOptions.Builder()
                .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                .build()
        )
    }

    /**
     * Decodes a QR code anywhere in [bitmap]. Returns the raw text payload, or null if none
     * was found or decoding failed.
     */
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

    /**
     * Decodes the QR straight from the original, unprocessed camera frame — not the warped
     * and denoised standardized card image. This matters: the standardized image has been
     * through perspective-warp interpolation, CLAHE, and a 3x3 median blur, all tuned for
     * reading bubble fill rather than preserving the sharp module edges a QR decoder needs,
     * and at ~160x180px for the whole QR wrapper that's often enough to break decoding
     * entirely. This projects Templates.SHARED_QR_CORNERS into the frame via the same
     * card-corners homography used elsewhere in the pipeline, crops around it with padding,
     * and decodes that — full camera resolution, no denoising, no interpolation.
     *
     * Falls back to [decodeQrRegion] on the standardized bitmap if this doesn't find anything
     * (e.g. corners were slightly off, or the crop missed the code).
     */
    suspend fun decodeFromOriginalFrame(
        frameBitmap: Bitmap,
        cardCorners: List<PointF>,
        standardizedFallback: Bitmap? = null
    ): String? {
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
                if (result != null) return result
            }
        }

        // Fall back to a full scan of the raw frame — slower (larger image) but still far
        // sharper than the standardized image, and doesn't depend on cardCorners being exact.
        decode(frameBitmap)?.let { return it }

        return standardizedFallback?.let { decodeQrRegion(it) }
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

    /**
     * Decodes only the known QR wrapper region (in Templates.REF_WIDTH x REF_HEIGHT
     * standardized card space, per Templates.SHARED_QR_CORNERS — the same rectangle for both
     * Candidate and Agenda cards, since template type is exactly what we're trying to
     * determine here). Prefer [decodeFromOriginalFrame] when the raw frame + cardCorners are
     * available; this is the fallback for when only the standardized bitmap is on hand.
     */
    suspend fun decodeQrRegion(standardizedBitmap: Bitmap): String? {
        val corners = Templates.SHARED_QR_CORNERS
        if (corners.size != 4) return decode(standardizedBitmap)

        decodeCroppedRegion(standardizedBitmap, corners, paddingFraction = 0.25f)?.let { return it }
        return decode(standardizedBitmap)
    }
}