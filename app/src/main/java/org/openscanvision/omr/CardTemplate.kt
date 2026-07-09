// CardTemplate.kt
package org.openscanvision.omr

import android.graphics.PointF

data class CardTemplate(
    val name: String,
    val prefix: String,
    val qrRefCorners: List<PointF>,
    val bubblePositions: List<PointF>,
    val markerRefPositions: List<PointF>? = null,
    val bubbleGroups: List<IntRange>? = null
)

object Templates {
    // 1 unit = 0.1 mm (85mm × 54mm = 850 × 540)
    const val REF_WIDTH = 850
    const val REF_HEIGHT = 540

    // QR code wrapper (detected corners stay the same as before)
    val SHARED_QR_CORNERS = listOf(
        PointF(592.5f, 52.5f),   // TL
        PointF(727.5f, 52.5f),   // TR
        PointF(727.5f, 187.5f),  // BR
        PointF(592.5f, 187.5f)   // BL
    )

    // 5mm × 5mm square markers at 1.2mm inset from the card edges
    val SHARED_MARKER_CORNERS = listOf(
        PointF(12f, 12f),      // TL
        PointF(838f, 12f),     // TR
        PointF(838f, 528f),    // BR
        PointF(12f, 528f)      // BL
    )

    // Candidate: single race, 3 columns × 4 rows
    val CANDIDATE_GROUPS = listOf(0..11)

    val CANDIDATE = CardTemplate(
        name = "Candidate",
        prefix = "VX",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CORNERS,
        bubbleGroups = CANDIDATE_GROUPS,
        bubblePositions = listOf(
            PointF(237f, 260f), PointF(470f, 260f), PointF(703f, 260f),
            PointF(237f, 320f), PointF(470f, 320f), PointF(703f, 320f),
            PointF(237f, 380f), PointF(470f, 380f), PointF(703f, 380f),
            PointF(237f, 440f), PointF(470f, 440f), PointF(703f, 440f)
        )
    )

    // Agenda: 4 rows, each with a label column + 3 option columns
    val AGENDA_GROUPS = listOf(0..2, 3..5, 6..8, 9..11)

    val AGENDA = CardTemplate(
        name = "Agenda",
        prefix = "AGN",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CORNERS,
        bubbleGroups = AGENDA_GROUPS,
        bubblePositions = listOf(
            PointF(450f, 260f), PointF(630f, 260f), PointF(810f, 260f),
            PointF(450f, 320f), PointF(630f, 320f), PointF(810f, 320f),
            PointF(450f, 380f), PointF(630f, 380f), PointF(810f, 380f),
            PointF(450f, 440f), PointF(630f, 440f), PointF(810f, 440f)
        )
    )

    fun fromPrefix(prefix: String): CardTemplate? =
        when {
            prefix.startsWith(CANDIDATE.prefix) -> CANDIDATE
            prefix.startsWith(AGENDA.prefix) -> AGENDA
            else -> null
        }
}