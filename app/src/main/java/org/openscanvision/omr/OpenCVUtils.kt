package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.PointF
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.atan2

object OpenCVUtils {

    /**
     * Detects the four card corners in the given [bitmap] with sub‑pixel precision.
     * Returns corners in order TL, TR, BR, BL, or null if no good quadrilateral is found.
     */
    fun detectCardCorners(bitmap: Bitmap): List<PointF>? {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // 1. Convert to grayscale and enhance edges
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)   // adjust thresholds for your environment

        // 2. Dilate to close small gaps
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.dilate(edges, edges, kernel)

        // 3. Find all external contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        // 4. Locate the largest quadrilateral
        var bestApprox: MatOfPoint2f? = null
        var maxArea = 0.0
        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < 1000) continue   // ignore noise
            val peri = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, 0.02 * peri, true)
            if (approx.total() == 4L && area > maxArea) {
                maxArea = area
                bestApprox = approx
            }
        }

        if (bestApprox == null) return null

        // 5. Sub‑pixel corner refinement
        val refined = MatOfPoint2f()
        bestApprox.copyTo(refined)
        Imgproc.cornerSubPix(
            gray, refined, Size(5.0, 5.0), Size(-1.0, -1.0),
            TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 40, 0.001)
        )

        // 6. Convert to PointF list and order TL, TR, BR, BL
        val pts = refined.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        return sortCornersTopLeftFirst(pts)
    }

    /**
     * Warps the card using OpenCV’s perspective transform.
     */
    fun warpCard(bitmap: Bitmap, corners: List<PointF>, dstWidth: Int, dstHeight: Int): Bitmap? {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val srcPoints = MatOfPoint2f()
        srcPoints.fromList(corners.map { Point(it.x.toDouble(), it.y.toDouble()) })

        val dstPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(dstWidth - 1.0, 0.0),
            Point(dstWidth - 1.0, dstHeight - 1.0),
            Point(0.0, dstHeight - 1.0)
        )

        val homography = Imgproc.getPerspectiveTransform(srcPoints, dstPoints)
        val warped = Mat()
        Imgproc.warpPerspective(src, warped, homography, Size(dstWidth.toDouble(), dstHeight.toDouble()))

        val result = Bitmap.createBitmap(dstWidth, dstHeight, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(warped, result)
        return result
    }

    /** Orders points into TL, TR, BR, BL using angle‑from‑centroid method. */
    private fun sortCornersTopLeftFirst(corners: List<PointF>): List<PointF> {
        if (corners.size != 4) return corners
        val cx = corners.map { it.x }.average().toFloat()
        val cy = corners.map { it.y }.average().toFloat()
        val sorted = corners.sortedBy { atan2((it.y - cy).toDouble(), (it.x - cx).toDouble()) }
        // After angle sort, the order is usually BL, TL, TR, BR – rotate to start from TL.
        val tlIndex = sorted.indexOfFirst { it.x <= cx && it.y <= cy }
        val rotated = if (tlIndex >= 0) sorted.drop(tlIndex) + sorted.take(tlIndex) else sorted
        return listOf(rotated[0], rotated[1], rotated[2], rotated[3])
    }
}