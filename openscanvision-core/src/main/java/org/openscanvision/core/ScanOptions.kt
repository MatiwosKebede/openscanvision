package org.openscanvision.core

import org.openscanvision.core.internal.omr.CardTemplate
import org.openscanvision.core.internal.omr.Templates

data class ScanOptions internal constructor(
    val template: CardTemplate,
    val confidenceThreshold: Float,
    val warpScale: Float,
    val enableClahe: Boolean,
    val medianBlurKernel: Int,
    val bubbleRadius: Int,
    val enableQrDecoding: Boolean,
    val requireQrMatch: Boolean,
    val generateAnnotatedImage: Boolean
) {
    class Builder {
        private var template: CardTemplate = Templates.CANDIDATE
        private var confidenceThreshold: Float = 0.6f
        private var warpScale: Float = 1.0f          // default to full resolution
        private var enableClahe: Boolean = true
        private var medianBlurKernel: Int = 3
        private var bubbleRadius: Int = 10
        private var enableQrDecoding: Boolean = true
        private var requireQrMatch: Boolean = false
        private var generateAnnotatedImage: Boolean = true

        fun template(template: CardTemplate) = apply { this.template = template }
        fun confidenceThreshold(threshold: Float) = apply { this.confidenceThreshold = threshold }
        fun warpScale(scale: Float) = apply { this.warpScale = scale }
        fun enableClahe(enabled: Boolean) = apply { this.enableClahe = enabled }
        fun medianBlurKernel(size: Int) = apply { this.medianBlurKernel = size }
        fun bubbleRadius(radius: Int) = apply { this.bubbleRadius = radius }
        fun enableQrDecoding(enabled: Boolean) = apply { this.enableQrDecoding = enabled }
        fun requireQrMatch(require: Boolean) = apply { this.requireQrMatch = require }
        fun generateAnnotatedImage(generate: Boolean) = apply { this.generateAnnotatedImage = generate }

        fun build() = ScanOptions(
            template, confidenceThreshold, warpScale,
            enableClahe, medianBlurKernel, bubbleRadius,
            enableQrDecoding, requireQrMatch, generateAnnotatedImage
        )
    }
    companion object {
        val Default = Builder().build()
    }
}