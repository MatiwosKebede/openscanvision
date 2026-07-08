package org.openscanvision.omr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

fun ImageProxy.toBitmap(): Bitmap? {
    val nv21 = yuv420888ToNv21(this) ?: return null
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val bytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}

private fun yuv420888ToNv21(image: ImageProxy): ByteArray? {
    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]

    val yBuffer = yPlane.buffer
    val uBuffer = uPlane.buffer
    val vBuffer = vPlane.buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)

    // NV21: Y plane then interleaved VU
    val uvSize = uSize
    val uvBuffer = ByteArray(uvSize * 2)
    var idx = 0
    for (i in 0 until uvSize) {
        uvBuffer[idx++] = vBuffer.get()
        uvBuffer[idx++] = uBuffer.get()
    }
    System.arraycopy(uvBuffer, 0, nv21, ySize, uvBuffer.size)
    return nv21
}
