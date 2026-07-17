// ScannerUtils.kt
package org.openscanvision.ui.utils

import android.graphics.Bitmap
import android.graphics.PointF
import androidx.camera.view.PreviewView
import androidx.compose.ui.geometry.Offset

fun mapImageToScreen(
    imagePoints: List<Offset>,
    imageWidth: Float,
    imageHeight: Float,
    previewView: PreviewView
): List<Offset>? {
    val viewWidth = previewView.width
    val viewHeight = previewView.height
    if (viewWidth == 0 || viewHeight == 0) return null

    val viewAspect = viewWidth.toFloat() / viewHeight
    val imageAspect = imageWidth / imageHeight

    val scaleX: Float; val scaleY: Float; val offsetX: Float; val offsetY: Float
    if (viewAspect > imageAspect) {
        scaleY = viewHeight / imageHeight; scaleX = scaleY
        offsetX = (viewWidth - imageWidth * scaleX) / 2f; offsetY = 0f
    } else {
        scaleX = viewWidth / imageWidth; scaleY = scaleX
        offsetX = 0f; offsetY = (viewHeight - imageHeight * scaleY) / 2f
    }

    return imagePoints.map { pt -> Offset(pt.x * scaleX + offsetX, pt.y * scaleY + offsetY) }
}

fun downsampleBitmap(bitmap: Bitmap, maxWidth: Int = 800, maxHeight: Int = 450): Bitmap {
    val width = bitmap.width; val height = bitmap.height
    val scale = minOf(maxWidth.toFloat() / width, maxHeight.toFloat() / height, 1f)
    if (scale >= 1f) return bitmap
    return Bitmap.createScaledBitmap(bitmap, (width * scale).toInt(), (height * scale).toInt(), true)
}