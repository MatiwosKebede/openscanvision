package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.aruco.Aruco
import org.opencv.aruco.DetectorParameters
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

object CardDetector {
    private const val TAG = "CardDetector"

    // ArUco dictionary used on the cards
    private const val ARUCO_DICT_ID = Aruco.DICT_5X5_50

    // ─── Card edge detection (fallback when ArUco fails) ─────────

    private const val MIN_AREA = 15000
    private const val MAX_AREA_RATIO = 0.85
    private const val EPSILON_FACTOR = 0.02

    fun detectCardCorners(bitmap: Bitmap): List<PointF>? {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        if (src.empty()) return null

        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY)

        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(blurred, edges, 50.0, 150.0)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        val dilated = Mat()
        Imgproc.dilate(edges, dilated, kernel)
        kernel.release()

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(dilated, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imgArea = src.width() * src.height()
        var bestContour2f: MatOfPoint2f? = null
        var maxArea = 0.0

        for (contour in contours) {
            val area = Imgproc.contourArea(contour)
            if (area < MIN_AREA || area > imgArea * MAX_AREA_RATIO) continue

            val contour2f = MatOfPoint2f()
            contour2f.fromArray(*contour.toArray())
            val peri = Imgproc.arcLength(contour2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx, peri * EPSILON_FACTOR, true)
            contour2f.release()

            if (approx.rows() == 4 && area > maxArea) {
                maxArea = area
                bestContour2f = MatOfPoint2f()
                approx.copyTo(bestContour2f)
            }
            approx.release()
        }

        src.release(); gray.release(); blurred.release(); edges.release(); dilated.release(); hierarchy.release()
        contours.forEach { it.release() }

        bestContour2f ?: return null
        val points = bestContour2f!!.toArray().map { PointF(it.x.toFloat(), it.y.toFloat()) }
        bestContour2f!!.release()
        if (points.size != 4) return null

        val tl = points.minByOrNull { it.x + it.y }!!
        val br = points.maxByOrNull { it.x + it.y }!!
        val remaining = points.filter { it != tl && it != br }
        val tr = remaining.minByOrNull { it.x - it.y }!!
        val bl = remaining.maxByOrNull { it.x - it.y }!!

        return listOf(tl, tr, br, bl)
    }

    // ─── ArUco detection ─────────────────────────────────────────

    /**
     * Detects ArUco markers (IDs 0-3, DICT_5X5_50) with tuned parameters.
     * Tries full resolution first, then falls back to a downsampled image.
     */
    fun detectArUcoMarkers(bitmap: Bitmap): Map<Int, List<PointF>> {
        var result = detectArUcoMarkersInternal(bitmap)
        if (result.isNotEmpty()) return result

        val downsampled = Bitmap.createScaledBitmap(bitmap, 640, 480, true)
        result = detectArUcoMarkersInternal(downsampled)
        if (result.isNotEmpty()) {
            val scaleX = bitmap.width.toFloat() / downsampled.width
            val scaleY = bitmap.height.toFloat() / downsampled.height
            result = result.mapValues { (_, pts) ->
                pts.map { PointF(it.x * scaleX, it.y * scaleY) }
            }
        }
        if (downsampled != bitmap) downsampled.recycle()
        return result
    }

    private fun detectArUcoMarkersInternal(bitmap: Bitmap): Map<Int, List<PointF>> {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val gray = Mat()
        when (src.channels()) {
            1 -> src.copyTo(gray)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            else -> { src.release(); return emptyMap() }
        }
        src.release()

        val params = DetectorParameters.create()
        params.set_adaptiveThreshWinSizeMin(7)
        params.set_adaptiveThreshWinSizeMax(23)
        params.set_adaptiveThreshWinSizeStep(4)
        params.set_polygonalApproxAccuracyRate(0.08)
        params.set_minCornerDistanceRate(0.02)
        params.set_minMarkerPerimeterRate(0.01)
        params.set_minMarkerDistanceRate(0.02)
        params.set_perspectiveRemovePixelPerCell(2)
        params.set_perspectiveRemoveIgnoredMarginPerCell(0.1)

        val dictionary = Aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        val corners = ArrayList<Mat>()
        val ids = Mat()
        Aruco.detectMarkers(gray, dictionary, corners, ids, params)

        val result = mutableMapOf<Int, List<PointF>>()
        for (i in 0 until ids.rows()) {
            val id = ids.get(i, 0)[0].toInt()
            val cornerMat = corners[i]
            val pts = (0..3).map { j ->
                PointF(cornerMat.get(0, j)[0].toFloat(), cornerMat.get(0, j)[1].toFloat())
            }
            result[id] = pts
        }
        gray.release()
        return result
    }

    /**
     * Uses QR corners to locate each expected ArUco marker region, crops,
     * and detects the marker. Essential for small markers near card edges.
     */
    fun detectArUcoMarkersGuided(bitmap: Bitmap, qrCorners: List<PointF>): Map<Int, List<PointF>> {
        if (qrCorners.size != 4) return emptyMap()
        val homography = HomographySolver.solve(Templates.SHARED_QR_CORNERS, qrCorners) ?: return emptyMap()
        val scaleEstimate = bitmap.width.toFloat() / Templates.REF_WIDTH.toFloat()
        val markerCentres = Templates.SHARED_MARKER_CENTRES
        val markerSize = 50   // 5mm in 0.1mm units
        val padding = 15      // extra margin in 0.1mm

        val result = mutableMapOf<Int, List<PointF>>()
        for (i in markerCentres.indices) {
            val templatePt = markerCentres[i]
            val srcArr = floatArrayOf(templatePt.x, templatePt.y)
            val dstArr = FloatArray(2)
            homography.mapPoints(dstArr, srcArr)
            val imagePt = PointF(dstArr[0], dstArr[1])

            val half = (markerSize / 2f + padding) * scaleEstimate
            val left = (imagePt.x - half).toInt().coerceIn(0, bitmap.width - 1)
            val top = (imagePt.y - half).toInt().coerceIn(0, bitmap.height - 1)
            val right = (imagePt.x + half).toInt().coerceIn(0, bitmap.width - 1)
            val bottom = (imagePt.y + half).toInt().coerceIn(0, bitmap.height - 1)
            val w = right - left; val h = bottom - top
            if (w <= 0 || h <= 0) continue

            val roi = Bitmap.createBitmap(bitmap, left, top, w, h)
            val local = detectArUcoMarkersInRegion(roi, i)
            roi.recycle()
            if (local.containsKey(i)) {
                val corners = local[i]!!.map { PointF(it.x + left, it.y + top) }
                result[i] = corners
            }
        }
        return result
    }

    private fun detectArUcoMarkersInRegion(bitmap: Bitmap, markerId: Int): Map<Int, List<PointF>> {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val gray = Mat()
        when (src.channels()) {
            1 -> src.copyTo(gray)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            else -> { src.release(); return emptyMap() }
        }
        src.release()

        val params = DetectorParameters.create()
        params.set_adaptiveThreshWinSizeMin(3)
        params.set_adaptiveThreshWinSizeMax(23)
        params.set_adaptiveThreshWinSizeStep(2)
        params.set_polygonalApproxAccuracyRate(0.15)
        params.set_minCornerDistanceRate(0.01)
        params.set_minMarkerPerimeterRate(0.005)
        params.set_minMarkerDistanceRate(0.01)
        params.set_perspectiveRemovePixelPerCell(1)
        params.set_perspectiveRemoveIgnoredMarginPerCell(0.2)

        val dictionary = Aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        val corners = ArrayList<Mat>()
        val ids = Mat()
        Aruco.detectMarkers(gray, dictionary, corners, ids, params)

        val result = mutableMapOf<Int, List<PointF>>()
        for (i in 0 until ids.rows()) {
            val id = ids.get(i, 0)[0].toInt()
            val cornerMat = corners[i]
            val pts = (0..3).map { j ->
                PointF(cornerMat.get(0, j)[0].toFloat(), cornerMat.get(0, j)[1].toFloat())
            }
            result[id] = pts
        }
        gray.release()
        return result
    }

    /**
     * Builds a template‑to‑image homography from the four ArUco markers.
     */
    fun buildHomographyFromArUco(arUcoResult: Map<Int, List<PointF>>): Matrix? {
        if (arUcoResult.size < 4) {
            Log.w(TAG, "Not enough ArUco markers for homography (${arUcoResult.size}/4)")
            return null
        }

        val templatePoints = mutableListOf<PointF>()
        val imagePoints = mutableListOf<PointF>()

        for ((id, corners) in arUcoResult) {
            val templateCorners = Templates.ARUCO_TEMPLATE_CORNERS[id] ?: continue
            if (corners.size != 4 || templateCorners.size != 4) continue
            for (i in 0..3) {
                templatePoints.add(templateCorners[i])
                imagePoints.add(corners[i])
            }
        }

        if (templatePoints.size < 8) {
            Log.w(TAG, "Too few point pairs for homography: ${templatePoints.size}")
            return null
        }

        val src = MatOfPoint2f()
        val dst = MatOfPoint2f()
        templatePoints.forEach { src.push_back(MatOfPoint(Point(it.x.toDouble(), it.y.toDouble()))) }
        imagePoints.forEach { dst.push_back(MatOfPoint(Point(it.x.toDouble(), it.y.toDouble()))) }

        val hMat = try {
            Calib3d.findHomography(src, dst, 0)
        } catch (e: Exception) {
            Log.e(TAG, "findHomography failed", e)
            src.release(); dst.release()
            return null
        }
        src.release(); dst.release()

        if (hMat.empty()) {
            hMat.release()
            return null
        }

        val values = FloatArray(9)
        for (r in 0..2) {
            for (c in 0..2) {
                values[r * 3 + c] = hMat.get(r, c)[0].toFloat()
            }
        }
        hMat.release()

        val matrix = Matrix()
        matrix.setValues(values)
        return matrix
    }

    /**
     * Transforms template‑space points to image coordinates.
     */
    fun predictImagePoints(templatePoints: List<PointF>, homography: Matrix): List<PointF> {
        val srcArr = FloatArray(templatePoints.size * 2)
        templatePoints.forEachIndexed { i, pt ->
            srcArr[i * 2] = pt.x
            srcArr[i * 2 + 1] = pt.y
        }
        val dstArr = FloatArray(srcArr.size)
        homography.mapPoints(dstArr, srcArr)
        return templatePoints.indices.map { i ->
            PointF(dstArr[i * 2], dstArr[i * 2 + 1])
        }
    }

    // ─── Simple Homography Solver (used for guided region cropping) ───
    object HomographySolver {
        fun solve(src: List<PointF>, dst: List<PointF>): Matrix? {
            if (src.size != dst.size || src.size < 4) return null
            val (srcNorm, tSrc) = normalizePoints(src) ?: return null
            val (dstNorm, tDst) = normalizePoints(dst) ?: return null
            val n = srcNorm.size
            val a = Array(2 * n) { DoubleArray(8) }
            val b = DoubleArray(2 * n)
            for (i in 0 until n) {
                val x = srcNorm[i].x.toDouble()
                val y = srcNorm[i].y.toDouble()
                val xp = dstNorm[i].x.toDouble()
                val yp = dstNorm[i].y.toDouble()
                a[2 * i][0] = x; a[2 * i][1] = y; a[2 * i][2] = 1.0
                a[2 * i][3] = 0.0; a[2 * i][4] = 0.0; a[2 * i][5] = 0.0
                a[2 * i][6] = -x * xp; a[2 * i][7] = -y * xp
                b[2 * i] = xp
                a[2 * i + 1][0] = 0.0; a[2 * i + 1][1] = 0.0; a[2 * i + 1][2] = 0.0
                a[2 * i + 1][3] = x; a[2 * i + 1][4] = y; a[2 * i + 1][5] = 1.0
                a[2 * i + 1][6] = -x * yp; a[2 * i + 1][7] = -y * yp
                b[2 * i + 1] = yp
            }
            val h = solveLeastSquares(a, b, 2 * n, 8) ?: return null
            val hNorm = Matrix()
            hNorm.setValues(floatArrayOf(
                h[0].toFloat(), h[1].toFloat(), h[2].toFloat(),
                h[3].toFloat(), h[4].toFloat(), h[5].toFloat(),
                h[6].toFloat(), h[7].toFloat(), 1f
            ))
            val tDstInv = Matrix(); tDst.invert(tDstInv)
            val result = Matrix(tDstInv); result.preConcat(hNorm); result.preConcat(tSrc)
            return result
        }

        private fun normalizePoints(points: List<PointF>): Pair<List<PointF>, Matrix>? {
            val cx = points.map { it.x }.average().toFloat()
            val cy = points.map { it.y }.average().toFloat()
            val meanDist = points.map {
                kotlin.math.sqrt(((it.x - cx) * (it.x - cx) + (it.y - cy) * (it.y - cy)).toDouble())
            }.average()
            if (meanDist < 1e-6) return null
            val scale = (kotlin.math.sqrt(2.0) / meanDist).toFloat()
            val transform = Matrix().apply { postTranslate(-cx, -cy); postScale(scale, scale) }
            val srcArr = FloatArray(points.size * 2)
            points.forEachIndexed { i, p -> srcArr[i * 2] = p.x; srcArr[i * 2 + 1] = p.y }
            val dstArr = FloatArray(srcArr.size)
            transform.mapPoints(dstArr, srcArr)
            val norm = points.indices.map { i -> PointF(dstArr[i * 2], dstArr[i * 2 + 1]) }
            return Pair(norm, transform)
        }

        private fun solveLeastSquares(a: Array<DoubleArray>, b: DoubleArray, rows: Int, cols: Int): DoubleArray? {
            val ata = Array(cols) { DoubleArray(cols) }; val atb = DoubleArray(cols)
            for (i in 0 until cols) {
                for (j in 0 until cols) {
                    var sum = 0.0; for (k in 0 until rows) sum += a[k][i] * a[k][j]; ata[i][j] = sum
                }
                var sumB = 0.0; for (k in 0 until rows) sumB += a[k][i] * b[k]; atb[i] = sumB
            }
            return gaussianSolve(ata, atb, cols)
        }

        private fun gaussianSolve(a: Array<DoubleArray>, b: DoubleArray, n: Int): DoubleArray? {
            val aug = Array(n) { i -> DoubleArray(n + 1).also { row ->
                for (j in 0 until n) row[j] = a[i][j]; row[n] = b[i] } }
            for (col in 0 until n) {
                var pivot = col; var max = kotlin.math.abs(aug[col][col])
                for (r in col + 1 until n) if (kotlin.math.abs(aug[r][col]) > max) {
                    max = kotlin.math.abs(aug[r][col]); pivot = r }
                if (max < 1e-10) return null
                val tmp = aug[col]; aug[col] = aug[pivot]; aug[pivot] = tmp
                for (r in 0 until n) {
                    if (r == col) continue
                    val factor = aug[r][col] / aug[col][col]
                    for (c in col until n + 1) aug[r][c] -= factor * aug[col][c]
                }
            }
            return DoubleArray(n) { i -> aug[i][n] / aug[i][i] }
        }
    }
}