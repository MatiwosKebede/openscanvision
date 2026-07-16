package org.openscanvision.core.internal.omr

import android.graphics.Bitmap
import android.graphics.PointF
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

internal object OpenCVUtils {
    fun warpCard(
        original: Bitmap,
        corners: List<PointF>,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap? {
        if (corners.size != 4) return null
        val srcPoints = MatOfPoint2f(*corners.map { Point(it.x.toDouble(), it.y.toDouble()) }.toTypedArray())
        val dstPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(targetWidth.toDouble(), 0.0),
            Point(targetWidth.toDouble(), targetHeight.toDouble()),
            Point(0.0, targetHeight.toDouble())
        )
        val homography = Calib3d.findHomography(srcPoints, dstPoints)
        val src = Mat()
        Utils.bitmapToMat(original, src)
        val dst = Mat()
        Imgproc.warpPerspective(src, dst, homography, Size(targetWidth.toDouble(), targetHeight.toDouble()))
        val result = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dst, result)
        src.release(); dst.release(); homography.release()
        return result
    }
}
