package org.openscanvision.core.internal.omr

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.aruco.Aruco
import org.opencv.aruco.DetectorParameters
import org.opencv.aruco.Dictionary
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

 object CardDetector {
    private const val TAG = "CardDetector"

    private const val ARUCO_DICT_ID = Aruco.DICT_5X5_50
    private const val MIN_VALID_AREA = 15.0
    private const val MAX_ASPECT_RATIO = 2.8
    private const val MAX_REPROJ_ERROR_PX = 15.0

    private const val MARKER_SEARCH_MARGIN_PX = 50
    private const val MARKER_DARKNESS_THRESHOLD = 70
    private const val MARKER_MIN_DARK_PIXELS = 15

    private const val MIN_CARD_AREA = 15000
    private const val MAX_CARD_AREA_RATIO = 0.85
    private const val EPSILON_FACTOR = 0.02

    private const val PAIR_SCALE_MIN = 0.15f
    private const val PAIR_SCALE_MAX = 8.0f
    private const val SCALE_AGREEMENT_TOLERANCE = 0.30f
    private const val ANGLE_AGREEMENT_TOLERANCE_DEG = 18f

    private val sharedDictionary: Dictionary by lazy {
        Aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    }

    val fastParams: DetectorParameters by lazy {
        DetectorParameters.create().apply {
            set_adaptiveThreshWinSizeMin(3)
            set_adaptiveThreshWinSizeMax(15)
            set_adaptiveThreshWinSizeStep(8)
            set_polygonalApproxAccuracyRate(0.15)
            set_minCornerDistanceRate(0.02)
            set_minMarkerPerimeterRate(0.02)
            set_minMarkerDistanceRate(0.02)
            set_perspectiveRemovePixelPerCell(2)
            set_perspectiveRemoveIgnoredMarginPerCell(0.2)
            set_cornerRefinementMethod(Aruco.CORNER_REFINE_NONE)
        }
    }

    val trackingParams: DetectorParameters by lazy {
        DetectorParameters.create().apply {
            set_adaptiveThreshWinSizeMin(3)
            set_adaptiveThreshWinSizeMax(17)
            set_adaptiveThreshWinSizeStep(6)
            set_polygonalApproxAccuracyRate(0.08)
            set_minCornerDistanceRate(0.01)
            set_minMarkerPerimeterRate(0.006)
            set_minMarkerDistanceRate(0.015)
            set_perspectiveRemovePixelPerCell(4)
            set_perspectiveRemoveIgnoredMarginPerCell(0.15)
            set_cornerRefinementMethod(Aruco.CORNER_REFINE_SUBPIX)
            set_cornerRefinementWinSize(3)
            set_cornerRefinementMaxIterations(15)
            set_cornerRefinementMinAccuracy(0.05)
        }
    }

    val accurateParams: DetectorParameters by lazy {
        DetectorParameters.create().apply {
            set_adaptiveThreshWinSizeMin(3)
            set_adaptiveThreshWinSizeMax(31)
            set_adaptiveThreshWinSizeStep(6)
            set_polygonalApproxAccuracyRate(0.08)
            set_minCornerDistanceRate(0.01)
            set_minMarkerPerimeterRate(0.005)
            set_minMarkerDistanceRate(0.01)
            set_perspectiveRemovePixelPerCell(8)
            set_perspectiveRemoveIgnoredMarginPerCell(0.13)
            set_cornerRefinementMethod(Aruco.CORNER_REFINE_SUBPIX)
            set_cornerRefinementWinSize(7)
            set_cornerRefinementMaxIterations(50)
            set_cornerRefinementMinAccuracy(0.01)
        }
    }

    data class TrackedMarker(
        val id: Int,
        val corners: List<PointF>,
        val center: PointF,
        val velocity: PointF,
        val missedFrames: Int = 0
    )

    class ArUcoTracker {
        private val markers = mutableMapOf<Int, TrackedMarker>()
        private var lostFrames = 0

        val isTrackingGood: Boolean
            get() = markers.isNotEmpty() && lostFrames < 3

        fun reset() {
            markers.clear()
            lostFrames = 0
        }

        fun detect(gray: Mat): Map<Int, List<PointF>> {
            val detected = if (isTrackingGood) {
                val centres = markers.mapValues { (_, marker) ->
                    PointF(
                        marker.center.x + marker.velocity.x,
                        marker.center.y + marker.velocity.y
                    )
                }

                val sizes = markers.mapValues { (_, marker) ->
                    markerSearchHalfSize(marker.corners)
                }

                val roiResult = detectArUcoMarkersTrackedInGray(gray, centres, sizes)

                if (roiResult.isNotEmpty()) {
                    roiResult
                } else {
                    lostFrames++
                    if (lostFrames >= 3) detectArUcoMarkersReacquire(gray) else emptyMap()
                }
            } else {
                detectArUcoMarkersReacquire(gray)
            }

            if (detected.isNotEmpty()) {
                update(detected)
                lostFrames = 0
            }

            return markers.mapValues { it.value.corners }
        }

        private fun update(detected: Map<Int, List<PointF>>) {
            val newIds = detected.keys.toSet()

            for ((id, old) in markers.toMap()) {
                if (id !in newIds) {
                    markers[id] = old.copy(missedFrames = old.missedFrames + 1)
                }
            }

            markers.entries.removeAll { it.value.missedFrames > 5 }

            for ((id, currentCorners) in detected) {
                val currentCenter = centerOf(currentCorners)
                val previous = markers[id]

                if (previous == null) {
                    markers[id] = TrackedMarker(
                        id = id,
                        corners = currentCorners,
                        center = currentCenter,
                        velocity = PointF(0f, 0f)
                    )
                } else {
                    val movement = distance(previous.center, currentCenter)
                    val alpha = if (movement > 20f) 0.75f else 0.25f
                    val smoothCorners = smoothCorners(previous.corners, currentCorners, alpha)
                    val smoothCenter = centerOf(smoothCorners)

                    markers[id] = TrackedMarker(
                        id = id,
                        corners = smoothCorners,
                        center = smoothCenter,
                        velocity = PointF(
                            smoothCenter.x - previous.center.x,
                            smoothCenter.y - previous.center.y
                        ),
                        missedFrames = 0
                    )
                }
            }
        }
    }

    fun isMarkerValid(corners: List<PointF>): Boolean {
        if (corners.size != 4) return false

        var area = 0.0
        for (i in 0..3) {
            val j = (i + 1) % 4
            area += corners[i].x * corners[j].y - corners[j].x * corners[i].y
        }

        area = abs(area) / 2.0
        if (area < MIN_VALID_AREA) return false

        val sideLengths = (0..3).map { i ->
            val j = (i + 1) % 4
            val dx = corners[i].x - corners[j].x
            val dy = corners[i].y - corners[j].y
            sqrt((dx * dx + dy * dy).toDouble())
        }

        val minSide = sideLengths.minOrNull() ?: return false
        val maxSide = sideLengths.maxOrNull() ?: return false

        if (minSide < 1e-6 || maxSide / minSide > MAX_ASPECT_RATIO) return false

        val signs = mutableListOf<Float>()
        for (i in 0 until corners.size) {
            val p1 = corners[i]
            val p2 = corners[(i + 1) % 4]
            val p3 = corners[(i + 2) % 4]
            val cross = (p2.x - p1.x) * (p3.y - p1.y) -
                    (p2.y - p1.y) * (p3.x - p1.x)
            signs.add(cross)
        }

        return signs.all { it > 0f } || signs.all { it < 0f }
    }

    fun detectArUcoMarkersInGray(
        gray: Mat,
        params: DetectorParameters = trackingParams
    ): Map<Int, List<PointF>> {
        val corners = ArrayList<Mat>()
        val ids = Mat()

        try {
            Aruco.detectMarkers(gray, sharedDictionary, corners, ids, params)

            val result = mutableMapOf<Int, List<PointF>>()

            for (i in 0 until ids.rows()) {
                val id = ids.get(i, 0)[0].toInt()
                val cornerMat = corners[i]

                val pts = (0..3).map { j ->
                    val p = cornerMat.get(0, j)
                    PointF(p[0].toFloat(), p[1].toFloat())
                }

                if (isMarkerValid(pts)) {
                    result[id] = pts
                }
            }

            return result
        } finally {
            ids.release()
            corners.forEach { it.release() }
        }
    }

    fun detectArUcoMarkersTrackedInGray(
        gray: Mat,
        searchCentres: Map<Int, PointF>,
        halfSizeMap: Map<Int, Int>
    ): Map<Int, List<PointF>> {
        if (searchCentres.isEmpty()) return emptyMap()

        val result = mutableMapOf<Int, List<PointF>>()
        val width = gray.cols()
        val height = gray.rows()

        for ((id, centre) in searchCentres) {
            val halfSize = halfSizeMap[id] ?: 50
            val rect = roiAround(centre, halfSize, width, height) ?: continue

            val roi = Mat(gray, rect)
            val localMap = try {
                detectArUcoMarkersInGray(roi, trackingParams)
            } finally {
                roi.release()
            }

            val corners = localMap[id]
            if (corners != null && corners.size == 4) {
                result[id] = corners.map {
                    PointF(it.x + rect.x, it.y + rect.y)
                }
            }
        }

        return result
    }

    fun detectArUcoMarkersReacquire(
        gray: Mat,
        downscaleFactors: List<Double> = listOf(0.4, 1.0),
        earlyBreakCount: Int = 3
    ): Map<Int, List<PointF>> {
        var bestResult = emptyMap<Int, List<PointF>>()
        var bestCount = 0

        for (scale in downscaleFactors) {
            val small = Mat()

            val coarse = try {
                if (scale == 1.0) {
                    detectArUcoMarkersInGray(gray, trackingParams)
                } else {
                    Imgproc.resize(gray, small, Size(), scale, scale, Imgproc.INTER_AREA)
                    detectArUcoMarkersInGray(small, fastParams)
                }
            } finally {
                small.release()
            }

            if (coarse.isEmpty()) continue

            val invScale = 1.0 / scale
            val result = mutableMapOf<Int, List<PointF>>()

            for ((id, smallCorners) in coarse) {
                if (scale == 1.0) {
                    result[id] = smallCorners
                    continue
                }

                val fullCorners = smallCorners.map {
                    PointF((it.x * invScale).toFloat(), (it.y * invScale).toFloat())
                }

                val center = centerOf(fullCorners)
                val halfSize = markerSearchHalfSize(fullCorners).coerceAtLeast(45)
                val rect = roiAround(center, halfSize, gray.cols(), gray.rows())

                if (rect == null) {
                    result[id] = fullCorners
                    continue
                }

                val roi = Mat(gray, rect)
                val refined = try {
                    detectArUcoMarkersInGray(roi, trackingParams)
                } finally {
                    roi.release()
                }

                val refinedCorners = refined[id]
                result[id] = if (refinedCorners != null && refinedCorners.size == 4) {
                    refinedCorners.map { PointF(it.x + rect.x, it.y + rect.y) }
                } else {
                    fullCorners
                }
            }

            if (result.size > bestCount) {
                bestResult = result
                bestCount = result.size
                if (bestCount >= earlyBreakCount) break
            }
        }

        return bestResult
    }

    fun rejectOutliersWithHomography(
        markers: Map<Int, List<PointF>>,
        templatePoints: List<PointF>
    ): Map<Int, List<PointF>> {
        if (markers.size < 4) return markers

        val tPts = mutableListOf<PointF>()
        val iPts = mutableListOf<PointF>()
        val ids = mutableListOf<Int>()

        for ((id, corners) in markers) {
            val templatePoint = templatePoints.getOrNull(id) ?: continue
            tPts.add(templatePoint)
            iPts.add(centerOf(corners))
            ids.add(id)
        }

        if (tPts.size < 4) return markers

        val homography = HomographySolver.solve(tPts, iPts) ?: return markers
        val errors = HomographySolver.reprojectionErrors(homography, tPts, iPts)

        val validIds = ids.filterIndexed { index, _ ->
            errors[index] < MAX_REPROJ_ERROR_PX
        }

        return markers.filterKeys { it in validIds }
    }

    private fun applySimilarity(modelPos: PointF, transform: FloatArray): PointF {
        val angle = transform[0]; val scale = transform[1]; val tx = transform[2]; val ty = transform[3]
        return PointF(
            scale * (cos(angle) * modelPos.x - sin(angle) * modelPos.y) + tx,
            scale * (sin(angle) * modelPos.x + cos(angle) * modelPos.y) + ty
        )
    }

    private fun estimateSimilarityLS(model: Map<Int, PointF>, imageCentres: Map<Int, PointF>): FloatArray? {
        val ids = model.keys.intersect(imageCentres.keys).toList()
        if (ids.size < 2) return null

        val mcx = ids.map { model[it]!!.x }.average().toFloat()
        val mcy = ids.map { model[it]!!.y }.average().toFloat()
        val icx = ids.map { imageCentres[it]!!.x }.average().toFloat()
        val icy = ids.map { imageCentres[it]!!.y }.average().toFloat()

        var numReal = 0.0; var numImag = 0.0; var denom = 0.0
        for (id in ids) {
            val mx = (model[id]!!.x - mcx).toDouble(); val my = (model[id]!!.y - mcy).toDouble()
            val ix = (imageCentres[id]!!.x - icx).toDouble(); val iy = (imageCentres[id]!!.y - icy).toDouble()
            numReal += mx * ix + my * iy
            numImag += mx * iy - my * ix
            denom += mx * mx + my * my
        }
        if (denom < 1e-6) return null
        val cReal = (numReal / denom).toFloat()
        val cImag = (numImag / denom).toFloat()
        val scale = sqrt((cReal * cReal + cImag * cImag).toDouble()).toFloat()
        if (scale < 1e-4f) return null
        val rotation = atan2(cImag, cReal)
        val tx = icx - (cReal * mcx - cImag * mcy)
        val ty = icy - (cImag * mcx + cReal * mcy)
        return floatArrayOf(rotation, scale, tx, ty)
    }

    fun verifyAgainstModel(model: Map<Int, PointF>, imageCentres: Map<Int, PointF>): Map<Int, PointF> {
        if (imageCentres.size < 2) return imageCentres

        data class PairEstimate(val idA: Int, val idB: Int, val scale: Float, val angleDeg: Float)

        val ids = imageCentres.keys.toList()
        val estimates = mutableListOf<PairEstimate>()
        for (i in ids.indices) {
            for (j in i + 1 until ids.size) {
                val id1 = ids[i]; val id2 = ids[j]
                val m1 = model[id1] ?: continue; val m2 = model[id2] ?: continue
                val modelDist = sqrt(((m2.x - m1.x) * (m2.x - m1.x) + (m2.y - m1.y) * (m2.y - m1.y)).toDouble()).toFloat()
                if (modelDist < 1e-3f) continue
                val p1 = imageCentres[id1]!!; val p2 = imageCentres[id2]!!
                val dx = p2.x - p1.x; val dy = p2.y - p1.y
                val imgDist = sqrt((dx * dx + dy * dy).toDouble()).toFloat()
                if (imgDist < 1e-3f) continue
                val scale = imgDist / modelDist
                val modelAngle = atan2((m2.y - m1.y).toDouble(), (m2.x - m1.x).toDouble())
                var angleDiffDeg = Math.toDegrees(atan2(dy.toDouble(), dx.toDouble()) - modelAngle).toFloat()
                angleDiffDeg = ((angleDiffDeg + 180f) % 360f + 360f) % 360f - 180f
                estimates.add(PairEstimate(id1, id2, scale, angleDiffDeg))
            }
        }
        if (estimates.isEmpty()) return imageCentres

        if (imageCentres.size == 2) {
            val est = estimates.first()
            return if (est.scale in PAIR_SCALE_MIN..PAIR_SCALE_MAX) imageCentres else emptyMap()
        }

        val medianScale = estimates.map { it.scale }.sorted()[estimates.size / 2]
        val medianAngle = estimates.map { it.angleDeg }.sorted()[estimates.size / 2]
        val agree = mutableMapOf<Int, Int>(); val disagree = mutableMapOf<Int, Int>()
        for (est in estimates) {
            val ok = est.scale in PAIR_SCALE_MIN..PAIR_SCALE_MAX &&
                    abs(est.scale - medianScale) <= medianScale * SCALE_AGREEMENT_TOLERANCE &&
                    abs(est.angleDeg - medianAngle) <= ANGLE_AGREEMENT_TOLERANCE_DEG
            for (id in listOf(est.idA, est.idB)) {
                if (ok) agree[id] = (agree[id] ?: 0) + 1 else disagree[id] = (disagree[id] ?: 0) + 1
            }
        }
        return imageCentres.filterKeys { id -> (agree[id] ?: 0) >= (disagree[id] ?: 0) }
    }

    fun predictMissingMarkerCentres(
        knownCentres: Map<Int, PointF>,
        fullModel: Map<Int, PointF> = Templates.SHARED_MARKER_CENTRES.withIndex().associate { (i, p) -> i to p }
    ): Map<Int, PointF> {
        val verified = verifyAgainstModel(fullModel, knownCentres)
        if (verified.size < 2) return emptyMap()
        val transform = estimateSimilarityLS(fullModel, verified) ?: return emptyMap()
        val predictions = mutableMapOf<Int, PointF>()
        for ((id, modelPos) in fullModel) {
            if (id !in verified) predictions[id] = applySimilarity(modelPos, transform)
        }
        return predictions
    }

    fun detectArUcoMarkersFull(bitmap: Bitmap): Map<Int, List<PointF>> {
        val gray = bitmapToGrayMat(bitmap)

        return try {
            detectArUcoMarkersInGray(gray, trackingParams)
        } finally {
            gray.release()
        }
    }

    fun detectArUcoMarkersGuided(
        bitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate
    ): Map<Int, List<PointF>> {
        val homography = HomographySolver.solve(template.qrRefCorners, qrCorners)
            ?: return emptyMap()

        val markerCentres = template.markerRefPositions ?: Templates.SHARED_MARKER_CENTRES
        val predictedCentres = predictImagePoints(markerCentres, homography)

        val scaleEstimate = bitmap.width.toFloat() / Templates.REF_WIDTH.toFloat()
        val halfSize = (70f * scaleEstimate).toInt().coerceAtLeast(60)

        val gray = bitmapToGrayMat(bitmap)

        try {
            val result = mutableMapOf<Int, List<PointF>>()

            for (i in markerCentres.indices) {
                val rect = roiAround(predictedCentres[i], halfSize, bitmap.width, bitmap.height)
                    ?: continue

                val roi = Mat(gray, rect)
                val localMap = try {
                    detectArUcoMarkersInGray(roi, trackingParams)
                } finally {
                    roi.release()
                }

                val markerCorners = localMap[i]
                if (markerCorners != null && markerCorners.size == 4) {
                    result[i] = markerCorners.map {
                        PointF(it.x + rect.x, it.y + rect.y)
                    }
                }
            }

            return result
        } finally {
            gray.release()
        }
    }

    fun buildHomographyFromArUco(arUcoResult: Map<Int, List<PointF>>): Matrix? {
        if (arUcoResult.size < 2) return null

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

        if (templatePoints.size < 8) return null

        return findHomographyOpenCv(
            srcPoints = templatePoints,
            dstPoints = imagePoints,
            ransacThreshold = 5.0
        )
    }

    fun computeArUcoHomographyError(arUcoResult: Map<Int, List<PointF>>): Float? {
        if (arUcoResult.size < 2) return null
        val homography = buildHomographyFromArUco(arUcoResult) ?: return null

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
        if (templatePoints.size < 8) return null

        val errors = HomographySolver.reprojectionErrors(homography, templatePoints, imagePoints)
        return if (errors.isNotEmpty()) errors.average().toFloat() else null
    }

    fun computeHomographyWithMarkers(
        bitmap: Bitmap,
        qrCorners: List<PointF>,
        template: CardTemplate
    ): Pair<Matrix, Float>? {
        if (qrCorners.size != 4) return null

        val qrHomography = HomographySolver.solve(template.qrRefCorners, qrCorners)
            ?: return null

        var arUcoMap = detectArUcoMarkersGuided(bitmap, qrCorners, template)

        if (arUcoMap.size < 2) {
            arUcoMap = detectArUcoMarkersFull(bitmap)
        }

        if (arUcoMap.size >= 2) {
            val arucoHomography = buildHomographyFromArUco(arUcoMap)

            if (arucoHomography != null) {
                val errors = HomographySolver.reprojectionErrors(
                    arucoHomography,
                    template.qrRefCorners,
                    qrCorners
                )

                val meanErr = errors.average().toFloat()

                if (meanErr < 10f) {
                    return Pair(arucoHomography, meanErr)
                }
            }
        }

        val markerRefs = template.markerRefPositions ?: Templates.SHARED_MARKER_CENTRES
        val templatePts = mutableListOf<PointF>()
        val imagePts = mutableListOf<PointF>()

        for (i in markerRefs.indices) {
            val corners = arUcoMap[i]
            if (corners != null && corners.size == 4) {
                templatePts.add(markerRefs[i])
                imagePts.add(centerOf(corners))
            }
        }

        if (templatePts.size < 2) {
            val predicted = predictImagePoints(markerRefs, qrHomography)
            val detected = detectMarkersNearPredicted(bitmap, predicted)

            for (i in markerRefs.indices) {
                val det = detected.getOrNull(i)
                if (det != null) {
                    templatePts.add(markerRefs[i])
                    imagePts.add(det)
                }
            }
        }

        if (templatePts.size >= 2) {
            templatePts.addAll(template.qrRefCorners)
            imagePts.addAll(qrCorners)

            val refined = computeRobustHomography(
                templatePoints = templatePts,
                imagePoints = imagePts,
                maxReprojErrorPx = 8f
            )

            if (refined != null) return refined
        }

        return Pair(qrHomography, 12f)
    }

    fun detectMarkersNearPredicted(
        bitmap: Bitmap,
        predictedPositions: List<PointF>
    ): List<PointF?> {
        val results = mutableListOf<PointF?>()

        for (pred in predictedPositions) {
            val left = (pred.x - MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(0, bitmap.width - 1)
            val top = (pred.y - MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(0, bitmap.height - 1)
            val right = (pred.x + MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(left + 1, bitmap.width)
            val bottom = (pred.y + MARKER_SEARCH_MARGIN_PX).toInt().coerceIn(top + 1, bitmap.height)

            val w = right - left
            val h = bottom - top

            if (w <= 0 || h <= 0) {
                results.add(null)
                continue
            }

            val pixels = IntArray(w * h)
            bitmap.getPixels(pixels, 0, w, left, top, w, h)

            var sumX = 0.0
            var sumY = 0.0
            var count = 0

            for (y in 0 until h) {
                for (x in 0 until w) {
                    val red = android.graphics.Color.red(pixels[y * w + x])
                    if (red < MARKER_DARKNESS_THRESHOLD) {
                        sumX += x
                        sumY += y
                        count++
                    }
                }
            }

            results.add(
                if (count >= MARKER_MIN_DARK_PIXELS) {
                    PointF(
                        (left + sumX / count).toFloat(),
                        (top + sumY / count).toFloat()
                    )
                } else {
                    null
                }
            )
        }

        return results
    }

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

    fun detectCardCorners(bitmap: Bitmap): List<PointF>? {
        val src = Mat()
        val gray = Mat()
        val blurred = Mat()
        val edges = Mat()
        val dilated = Mat()
        val hierarchy = Mat()
        val contours = mutableListOf<MatOfPoint>()

        try {
            Utils.bitmapToMat(bitmap, src)
            if (src.empty()) return null

            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGB2GRAY)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 50.0, 150.0)

            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                Size(5.0, 5.0)
            )

            try {
                Imgproc.dilate(edges, dilated, kernel)
            } finally {
                kernel.release()
            }

            Imgproc.findContours(
                dilated,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )

            val imgArea = src.width() * src.height()
            var bestContour: MatOfPoint2f? = null
            var maxArea = 0.0

            for (contour in contours) {
                val area = Imgproc.contourArea(contour)
                if (area < MIN_CARD_AREA || area > imgArea * MAX_CARD_AREA_RATIO) continue

                val contour2f = MatOfPoint2f()
                val approx = MatOfPoint2f()

                try {
                    contour2f.fromArray(*contour.toArray())
                    val peri = Imgproc.arcLength(contour2f, true)
                    Imgproc.approxPolyDP(contour2f, approx, peri * EPSILON_FACTOR, true)

                    if (approx.rows() == 4 && area > maxArea) {
                        bestContour?.release()
                        bestContour = MatOfPoint2f()
                        approx.copyTo(bestContour)
                        maxArea = area
                    }
                } finally {
                    contour2f.release()
                    approx.release()
                }
            }

            val points = bestContour?.toArray()?.map {
                PointF(it.x.toFloat(), it.y.toFloat())
            }

            bestContour?.release()

            if (points == null || points.size != 4) return null

            val tl = points.minByOrNull { it.x + it.y } ?: return null
            val br = points.maxByOrNull { it.x + it.y } ?: return null
            val remaining = points.filter { it != tl && it != br }

            if (remaining.size != 2) return null

            val tr = remaining.minByOrNull { it.x - it.y } ?: return null
            val bl = remaining.maxByOrNull { it.x - it.y } ?: return null

            return listOf(tl, tr, br, bl)
        } finally {
            src.release()
            gray.release()
            blurred.release()
            edges.release()
            dilated.release()
            hierarchy.release()
            contours.forEach { it.release() }
        }
    }

    private fun bitmapToGrayMat(bitmap: Bitmap): Mat {
        val src = Mat()
        val gray = Mat()

        Utils.bitmapToMat(bitmap, src)

        when (src.channels()) {
            1 -> src.copyTo(gray)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            else -> {}
        }

        src.release()
        return gray
    }

    private fun roiAround(center: PointF, halfSize: Int, width: Int, height: Int): Rect? {
        if (width <= 1 || height <= 1) return null

        val left = (center.x - halfSize).toInt().coerceIn(0, width - 1)
        val top = (center.y - halfSize).toInt().coerceIn(0, height - 1)
        val right = (center.x + halfSize).toInt().coerceIn(left + 1, width)
        val bottom = (center.y + halfSize).toInt().coerceIn(top + 1, height)

        val w = right - left
        val h = bottom - top

        if (w < 15 || h < 15) return null

        return Rect(left, top, w, h)
    }

    private fun markerSearchHalfSize(corners: List<PointF>): Int {
        if (corners.size != 4) return 50

        var total = 0f
        for (i in 0..3) {
            val a = corners[i]
            val b = corners[(i + 1) % 4]
            total += distance(a, b)
        }

        return ((total / 4f) * 2.5f).toInt().coerceIn(35, 180)
    }

    private fun centerOf(corners: List<PointF>): PointF {
        if (corners.isEmpty()) return PointF(0f, 0f)
        var x = 0f
        var y = 0f
        for (corner in corners) {
            x += corner.x
            y += corner.y
        }
        val count = corners.size.toFloat()
        return PointF(
            x / count,
            y / count
        )
    }

    private fun distance(a: PointF, b: PointF): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun smoothCorners(
        previous: List<PointF>,
        current: List<PointF>,
        alpha: Float
    ): List<PointF> {
        if (previous.size != 4 || current.size != 4) return current

        return current.indices.map { i ->
            PointF(
                previous[i].x * (1f - alpha) + current[i].x * alpha,
                previous[i].y * (1f - alpha) + current[i].y * alpha
            )
        }
    }

    private fun findHomographyOpenCv(
        srcPoints: List<PointF>,
        dstPoints: List<PointF>,
        ransacThreshold: Double
    ): Matrix? {
        if (srcPoints.size != dstPoints.size || srcPoints.size < 4) return null

        val src = MatOfPoint2f()
        val dst = MatOfPoint2f()

        return try {
            src.fromList(srcPoints.map { Point(it.x.toDouble(), it.y.toDouble()) })
            dst.fromList(dstPoints.map { Point(it.x.toDouble(), it.y.toDouble()) })

            val hMat = Calib3d.findHomography(
                src,
                dst,
                Calib3d.RANSAC,
                ransacThreshold
            )

            try {
                if (hMat.empty()) return null

                val values = FloatArray(9)
                for (r in 0..2) {
                    for (c in 0..2) {
                        values[r * 3 + c] = hMat.get(r, c)[0].toFloat()
                    }
                }

                Matrix().apply {
                    setValues(values)
                }
            } finally {
                hMat.release()
            }
        } catch (e: Exception) {
            null
        } finally {
            src.release()
            dst.release()
        }
    }

    private fun computeRobustHomography(
        templatePoints: List<PointF>,
        imagePoints: List<PointF>,
        maxReprojErrorPx: Float,
        minPoints: Int = 4
    ): Pair<Matrix, Float>? {
        if (templatePoints.size != imagePoints.size || templatePoints.size < minPoints) {
            return null
        }

        val cvHomography = findHomographyOpenCv(templatePoints, imagePoints, maxReprojErrorPx.toDouble())
        if (cvHomography != null) {
            val errors = HomographySolver.reprojectionErrors(cvHomography, templatePoints, imagePoints)
            val meanErr = if (errors.isNotEmpty()) errors.average().toFloat() else Float.MAX_VALUE
            return Pair(cvHomography, meanErr)
        }

        var curTmpl = templatePoints.toMutableList()
        var curImg = imagePoints.toMutableList()
        var homography = HomographySolver.solve(curTmpl, curImg) ?: return null

        while (curTmpl.size > minPoints) {
            val errors = HomographySolver.reprojectionErrors(homography, curTmpl, curImg)
            val maxErr = errors.maxOrNull() ?: break

            if (maxErr <= maxReprojErrorPx) break

            val worstIdx = errors.indexOf(maxErr)
            curTmpl.removeAt(worstIdx)
            curImg.removeAt(worstIdx)

            homography = HomographySolver.solve(curTmpl, curImg) ?: break
        }

        val finalErrors = HomographySolver.reprojectionErrors(homography, curTmpl, curImg)
        val meanErr = if (finalErrors.isNotEmpty()) {
            finalErrors.average().toFloat()
        } else {
            Float.MAX_VALUE
        }

        return Pair(homography, meanErr)
    }

    object HomographySolver {
        fun solve(src: List<PointF>, dst: List<PointF>): Matrix? {
            if (src.size != dst.size || src.size < 4) return null

            val src2f = MatOfPoint2f()
            val dst2f = MatOfPoint2f()

            return try {
                src2f.fromList(src.map { Point(it.x.toDouble(), it.y.toDouble()) })
                dst2f.fromList(dst.map { Point(it.x.toDouble(), it.y.toDouble()) })

                val hMat = Calib3d.findHomography(src2f, dst2f, 0)

                try {
                    if (hMat.empty()) return null

                    val values = FloatArray(9)
                    for (r in 0..2) {
                        for (c in 0..2) {
                            values[r * 3 + c] = hMat.get(r, c)[0].toFloat()
                        }
                    }

                    Matrix().apply {
                        setValues(values)
                    }
                } finally {
                    hMat.release()
                }
            } catch (e: Exception) {
                null
            } finally {
                src2f.release()
                dst2f.release()
            }
        }

        fun reprojectionErrors(
            homography: Matrix,
            templatePoints: List<PointF>,
            imagePoints: List<PointF>
        ): List<Float> {
            if (templatePoints.size != imagePoints.size) return emptyList()

            val srcArr = FloatArray(templatePoints.size * 2)

            templatePoints.forEachIndexed { i, pt ->
                srcArr[i * 2] = pt.x
                srcArr[i * 2 + 1] = pt.y
            }

            val dstArr = FloatArray(srcArr.size)
            homography.mapPoints(dstArr, srcArr)

            return imagePoints.indices.map { i ->
                val dx = dstArr[i * 2] - imagePoints[i].x
                val dy = dstArr[i * 2 + 1] - imagePoints[i].y
                sqrt(dx * dx + dy * dy)
            }
        }
    }
}
