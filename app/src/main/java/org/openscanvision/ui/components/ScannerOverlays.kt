// ScannerOverlays.kt
package org.openscanvision.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp


// ─── Smooth corner tracker ──────────────────────────────────────────
@Composable
fun rememberSmoothCorners(
    rawCorners: List<Offset>?,
    smoothing: Float = 0.3f
): List<Offset>? {
    var smoothed by remember { mutableStateOf<List<Offset>?>(null) }
    LaunchedEffect(rawCorners) {
        if (rawCorners == null) { smoothed = null; return@LaunchedEffect }
        val current = smoothed
        if (current == null || current.size != rawCorners.size) {
            smoothed = rawCorners
        } else {
            smoothed = current.indices.map { i ->
                val raw = rawCorners[i]
                val prev = current[i]
                Offset(
                    x = prev.x + (raw.x - prev.x) * smoothing,
                    y = prev.y + (raw.y - prev.y) * smoothing
                )
            }
        }
    }
    return smoothed
}

// ─── Live Card Overlay ──────────────────────────────────────────────
@Composable
fun LiveCardOverlay(
    cardCorners: List<Offset>?,
    qrCorners: List<Offset>?,
    markerCenters: Map<String, Offset>?,
    bubblePositions: List<Offset>?,
    bubbleStatus: List<Boolean>?,
    isTracking: Boolean,
    qrDetected: Boolean,
    scaleFactor: Float = 1.08f
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        // QR highlight
        if (qrDetected && qrCorners != null && qrCorners.size == 4) {
            val path = Path().apply {
                moveTo(qrCorners[0].x, qrCorners[0].y)
                lineTo(qrCorners[1].x, qrCorners[1].y)
                lineTo(qrCorners[2].x, qrCorners[2].y)
                lineTo(qrCorners[3].x, qrCorners[3].y)
                close()
            }
            drawPath(
                path = path,
                color = Color.Magenta.copy(alpha = 0.7f),
                style = Stroke(width = 3f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 6f)))
            )
            val cx = qrCorners.map { it.x }.average().toFloat()
            val cy = qrCorners.map { it.y }.average().toFloat()
            drawCircle(color = Color.Magenta.copy(alpha = 0.5f), radius = 12.dp.toPx(), center = Offset(cx, cy))
        }

        // Card outline
        if (cardCorners != null && cardCorners.size == 4) {
            val scaledCorners = if (scaleFactor != 1f) {
                val cx = cardCorners.map { it.x }.average().toFloat()
                val cy = cardCorners.map { it.y }.average().toFloat()
                cardCorners.map { corner ->
                    Offset(
                        cx + (corner.x - cx) * scaleFactor,
                        cy + (corner.y - cy) * scaleFactor
                    )
                }
            } else {
                cardCorners
            }

            val path = Path().apply {
                moveTo(scaledCorners[0].x, scaledCorners[0].y)
                lineTo(scaledCorners[1].x, scaledCorners[1].y)
                lineTo(scaledCorners[2].x, scaledCorners[2].y)
                lineTo(scaledCorners[3].x, scaledCorners[3].y)
                close()
            }
            drawPath(
                path = path,
                color = if (isTracking) Color.Green else Color(0xFFFEB914),
                style = Stroke(width = 4f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 8f)))
            )
            scaledCorners.forEach { corner ->
                drawCircle(color = Color.Cyan, radius = 8.dp.toPx(), center = corner)
            }
        }

        // Reference markers
        if (markerCenters != null) {
            val markerRadius = 8.dp.toPx()
            val markerColors = mapOf(
                "TL" to Color.Magenta,
                "BL" to Color.Magenta,
                "BR" to Color.Magenta
            )
            markerCenters.forEach { (label, center) ->
                val color = markerColors[label] ?: Color.Magenta
                drawCircle(
                    color = color.copy(alpha = 0.25f),
                    radius = markerRadius * 2.5f,
                    center = center
                )
                drawCircle(
                    color = color.copy(alpha = 0.9f),
                    radius = markerRadius,
                    center = center,
                    style = Stroke(width = 2.5f)
                )
                val crossSize = 10.dp.toPx()
                drawLine(
                    color = color.copy(alpha = 0.8f),
                    start = Offset(center.x - crossSize, center.y),
                    end = Offset(center.x + crossSize, center.y),
                    strokeWidth = 2.5f
                )
                drawLine(
                    color = color.copy(alpha = 0.8f),
                    start = Offset(center.x, center.y - crossSize),
                    end = Offset(center.x, center.y + crossSize),
                    strokeWidth = 2.5f
                )
                drawCircle(
                    color = color.copy(alpha = 0.5f),
                    radius = 4.dp.toPx(),
                    center = center
                )
            }
        }

        // Bubbles
        if (bubblePositions != null) {
            val bubbleRadius = 12.dp.toPx()
            val statusList = bubbleStatus ?: emptyList()

            bubblePositions.forEachIndexed { index, pos ->
                val status = statusList.getOrNull(index)
                val color = when (status) {
                    true -> Color.Green.copy(alpha = 0.9f)
                    false -> Color(0xFF3B82F6).copy(alpha = 0.7f)
                    else -> Color(0xFFFEB914).copy(alpha = 0.4f)
                }
                val glow = when (status) {
                    true -> Color.Green.copy(alpha = 0.3f)
                    false -> Color(0xFF3B82F6).copy(alpha = 0.2f)
                    else -> Color.Transparent
                }
                if (glow != Color.Transparent) {
                    drawCircle(
                        color = glow,
                        radius = bubbleRadius * 2.5f,
                        center = pos,
                        style = Stroke(width = 6f)
                    )
                }
                drawCircle(
                    color = color,
                    radius = bubbleRadius,
                    center = pos,
                    style = Stroke(width = 2f)
                )
                val cross = 6.dp.toPx()
                drawLine(
                    color = color,
                    start = Offset(pos.x - cross, pos.y),
                    end = Offset(pos.x + cross, pos.y),
                    strokeWidth = 2f
                )
                drawLine(
                    color = color,
                    start = Offset(pos.x, pos.y - cross),
                    end = Offset(pos.x, pos.y + cross),
                    strokeWidth = 2f
                )
            }
        }
    }
}

// ─── Static Viewfinder ──────────────────────────────────────────────
@Composable
fun StaticViewfinder() {
    val density = LocalDensity.current
    val config = LocalConfiguration.current
    val screenWidth = with(density) { config.screenWidthDp.dp.toPx() }
    val cardAspectRatio = 85f / 54f
    val viewfinderWidthPx = screenWidth * 0.92f
    val viewfinderHeightPx = viewfinderWidthPx / cardAspectRatio

    Box(modifier = Modifier.fillMaxSize()) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val cardLeft = (canvasWidth - viewfinderWidthPx) / 2f
            val cardTop = (canvasHeight - viewfinderHeightPx) / 2f
            val cardRight = cardLeft + viewfinderWidthPx
            val cardBottom = cardTop + viewfinderHeightPx
            val cornerRadius = 12.dp.toPx()

            val overlayPath = Path().apply {
                addRect(Rect(0f, 0f, canvasWidth, canvasHeight))
                addRoundRect(
                    RoundRect(
                        rect = Rect(cardLeft, cardTop, cardRight, cardBottom),
                        cornerRadius = CornerRadius(cornerRadius, cornerRadius)
                    )
                )
            }
            drawPath(overlayPath, Color.Black.copy(alpha = 0.55f), blendMode = BlendMode.SrcOver)

            val bracketLen = 36.dp.toPx()
            val bracketWidth = 3.5f
            val color = Color(0xFFFEB914)
            val vfRight = cardLeft + viewfinderWidthPx
            val vfBottom = cardTop + viewfinderHeightPx

            drawLine(color, Offset(cardLeft, cardTop + cornerRadius), Offset(cardLeft, cardTop + bracketLen), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(cardLeft + cornerRadius, cardTop), Offset(cardLeft + bracketLen, cardTop), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(vfRight, cardTop + cornerRadius), Offset(vfRight, cardTop + bracketLen), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(vfRight - cornerRadius, cardTop), Offset(vfRight - bracketLen, cardTop), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(cardLeft, vfBottom - cornerRadius), Offset(cardLeft, vfBottom - bracketLen), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(cardLeft + cornerRadius, vfBottom), Offset(cardLeft + bracketLen, vfBottom), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(vfRight, vfBottom - cornerRadius), Offset(vfRight, vfBottom - bracketLen), bracketWidth, StrokeCap.Round)
            drawLine(color, Offset(vfRight - cornerRadius, vfBottom), Offset(vfRight - bracketLen, vfBottom), bracketWidth, StrokeCap.Round)
        }

        Text(
            text = "Align card inside the frame",
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            color = Color.White.copy(alpha = 0.8f),
            textAlign = TextAlign.Center,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 40.dp)
        )
    }
}

// ─── Scanning Line Overlay ──────────────────────────────────────────
@Composable
fun ScanningLineOverlay(isScanning: Boolean, status: String, frames: Int, totalFrames: Int) {
    val transition = rememberInfiniteTransition()
    val scanProgress by transition.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(animation = tween(1500, easing = LinearEasing), repeatMode = RepeatMode.Reverse)
    )

    val density = LocalDensity.current
    val config = LocalConfiguration.current
    val screenWidth = with(density) { config.screenWidthDp.dp.toPx() }
    val cardAspectRatio = 85f / 54f
    val viewfinderWidthPx = screenWidth * 0.92f
    val viewfinderHeightPx = viewfinderWidthPx / cardAspectRatio

    Box(modifier = Modifier.fillMaxSize()) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height
            val cardLeft = (canvasWidth - viewfinderWidthPx) / 2f
            val cardTop = (canvasHeight - viewfinderHeightPx) / 2f
            val cardRight = cardLeft + viewfinderWidthPx

            val lineY = cardTop + (viewfinderHeightPx * scanProgress)
            val lineStartX = cardLeft + 20.dp.toPx()
            val lineEndX = cardRight - 20.dp.toPx()

            drawLine(color = Color(0xFFFEB914).copy(alpha = 0.8f), start = Offset(lineStartX, lineY), end = Offset(lineEndX, lineY), strokeWidth = 3f, cap = StrokeCap.Round)
            drawLine(color = Color(0xFFFEB914).copy(alpha = 0.2f), start = Offset(lineStartX, lineY), end = Offset(lineEndX, lineY), strokeWidth = 12f, cap = StrokeCap.Round)
        }

        if (isScanning) {
            Column(modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 80.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(text = status, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Medium, textAlign = TextAlign.Center)
                if (frames > 0 && totalFrames > 0) {
                    LinearProgressIndicator(
                        progress = frames.toFloat() / totalFrames,
                        modifier = Modifier.fillMaxWidth(0.5f).height(4.dp),
                        color = Color(0xFFFEB914),
                        trackColor = Color.White.copy(alpha = 0.3f)
                    )
                }
            }
        }
    }
}