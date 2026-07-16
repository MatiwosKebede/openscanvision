package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.suspendCancellableCoroutine
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.coroutines.resume

/**
 * QR decoding via ML Kit barcode-scanning. All the extra work here (perspective
 * rectification, multiple fallback attempts) runs exactly once per capture attempt,
 * never inside the per-frame ArUco tracking loop — so none of it affects live
 * scanning responsiveness.
 */
object QrDecoder {
    private const val TAG = "QrDecoder"

    private const val RECTIFIED_SIZE = 360
    private const val RECTIFIED_MARGIN_FRACTION = 0.18f

    private val scanner by lazy {
        BarcodeScanning.getClient(
            BarcodeScannerOptions.Builder()
                .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                .build()
        )
    }

    /**
     * Decodes a QR code anywhere in [bitmap]. [rotationDegrees] should be the sensor
     * rotation (0/90/180/270) if [bitmap] hasn't already been rotated to upright.
     */
    suspend fun decode(bitmap: Bitmap, rotationDegrees: Int = 0): String? {
        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360
        val image = InputImage.fromBitmap(bitmap, normalizedRotation)
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
     * Decodes the QR straight from the original, unprocessed camera frame — not the
     * warped and denoised standardized card image, which has been through
     * perspective-warp interpolation, CLAHE, and a median blur, all tuned for bubble
     * fill rather than preserving the sharp module edges a QR decoder needs.
     *
     * Tries, in order:
     *  1. A perspective-rectified crop (squares up skew/rotation, adds a clean quiet zone)
     *  2. An axis-aligned crop with generous padding
     *  3. A full-frame scan
     *  4. The standardized bitmap, if provided, as a last resort
     */
    suspend fun decodeFromOriginalFrame(
        frameBitmap: Bitmap,
        cardCorners: List<PointF>,
        standardizedFallback: Bitmap? = null,
        rotationDegrees: Int = 0
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

                val rectifiedResult = rectifyAndDecode(frameBitmap, qrCornersInFrame)
                if (rectifiedResult != null) return rectifiedResult

                val croppedResult = decodeCroppedRegion(frameBitmap, qrCornersInFrame, 0.5f, rotationDegrees)
                if (croppedResult != null) return croppedResult
            }
        }

        val fullFrameResult = decode(frameBitmap, rotationDegrees)
        if (fullFrameResult != null) return fullFrameResult

        return standardizedFallback?.let { decodeQrRegion(it) }
    }

    /**
     * Warps the frame so the QR's own corners land on an axis-aligned inset square,
     * undoing perspective skew/rotation and baking in a clean quiet zone in one step.
     * Uses OpenCV directly (same pattern as OMRExtractor's homography warp).
     */
    private suspend fun rectifyAndDecode(
        frameBitmap: Bitmap,
        qrCornersInFrame: List<PointF>
    ): String? {
        if (qrCornersInFrame.size != 4) return null

        val margin = RECTIFIED_SIZE * RECTIFIED_MARGIN_FRACTION
        val content = RECTIFIED_SIZE - 2 * margin

        val srcMat = MatOfPoint2f(
            Point(qrCornersInFrame[0].x.toDouble(), qrCornersInFrame[0].y.toDouble()),
            Point(qrCornersInFrame[1].x.toDouble(), qrCornersInFrame[1].y.toDouble()),
            Point(qrCornersInFrame[2].x.toDouble(), qrCornersInFrame[2].y.toDouble()),
            Point(qrCornersInFrame[3].x.toDouble(), qrCornersInFrame[3].y.toDouble())
        )
        val dstMat = MatOfPoint2f(
            Point(margin.toDouble(), margin.toDouble()),
            Point((margin + content).toDouble(), margin.toDouble()),
            Point((margin + content).toDouble(), (margin + content).toDouble()),
            Point(margin.toDouble(), (margin + content).toDouble())
        )

        val frameMat = Mat()
        val warpedMat = Mat()
        var homographyMat: Mat? = null
        var rectifiedBitmap: Bitmap? = null

        try {
            Utils.bitmapToMat(frameBitmap, frameMat)
            homographyMat = Imgproc.getPerspectiveTransform(srcMat, dstMat)
            Imgproc.warpPerspective(
                frameMat, warpedMat, homographyMat,
                Size(RECTIFIED_SIZE.toDouble(), RECTIFIED_SIZE.toDouble()),
                Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT,
                Scalar(255.0, 255.0, 255.0, 255.0)
            )
            if (!warpedMat.empty()) {
                val bmp = Bitmap.createBitmap(RECTIFIED_SIZE, RECTIFIED_SIZE, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(warpedMat, bmp)
                rectifiedBitmap = bmp
            }
        } catch (e: Exception) {
            Log.w(TAG, "QR rectification failed", e)
        } finally {
            frameMat.release()
            homographyMat?.release()
            warpedMat.release()
            srcMat.release()
            dstMat.release()
        }

        val finalBitmap = rectifiedBitmap ?: return null
        return decode(finalBitmap, 0)
    }

    private suspend fun decodeCroppedRegion(
        bitmap: Bitmap,
        corners: List<PointF>,
        paddingFraction: Float,
        rotationDegrees: Int = 0
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
            decode(crop, rotationDegrees)
        } finally {
            crop.recycle()
        }
    }

    /**
     * Decodes only the known QR wrapper region (in Templates.REF_WIDTH x REF_HEIGHT
     * standardized card space, per Templates.SHARED_QR_CORNERS). Prefer
     * [decodeFromOriginalFrame] when the raw frame + cardCorners are available; this
     * is the fallback for when only the standardized bitmap is on hand.
     */
    suspend fun decodeQrRegion(standardizedBitmap: Bitmap): String? {
        val corners = Templates.SHARED_QR_CORNERS
        if (corners.size != 4) return decode(standardizedBitmap)

        val croppedResult = decodeCroppedRegion(standardizedBitmap, corners, 0.25f)
        if (croppedResult != null) return croppedResult

        return decode(standardizedBitmap)
    }
}