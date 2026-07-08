// CardTemplate.kt
package org.openscanvision.omr

import android.graphics.PointF

data class CardTemplate(
    val name: String,
    val prefix: String,                       // "VX" for candidate, "AGN" for agenda
    val qrRefCorners: List<PointF>,            // 4 corners: TL, TR, BR, BL
    val bubblePositions: List<PointF>,         // bubble centers (x, y)
    val markerRefPositions: List<PointF>? = null,
    val bubbleGroups: List<IntRange>? = null
)

object Templates {
    // 1 unit = 0.1mm (matches 85mm x 54mm exactly)
    const val REF_WIDTH = 850
    const val REF_HEIGHT = 540

    val SHARED_QR_CORNERS = listOf(
        PointF(660f, 30f),   // TL
        PointF(820f, 30f),   // TR
        PointF(820f, 210f),  // BR
        PointF(660f, 210f)   // BL
    )

    val SHARED_MARKER_CORNERS = listOf(
        PointF(25f, 25f),         // TL
        PointF(825f, 515f),       // BR
        PointF(25f, 515f)         // BL
    )

    // Candidate: 3x4 grid, single race
    val CANDIDATE_GROUPS = listOf(0..11)

    val CANDIDATE = CardTemplate(
        name = "Candidate",
        prefix = "VX",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CORNERS,
        bubbleGroups = CANDIDATE_GROUPS,
        bubblePositions = listOf(
            PointF(58f, 260f), PointF(318f, 260f), PointF(578f, 260f),
            PointF(58f, 320f), PointF(318f, 320f), PointF(578f, 320f),
            PointF(58f, 380f), PointF(318f, 380f), PointF(578f, 380f),
            PointF(58f, 440f), PointF(318f, 440f), PointF(578f, 440f)
        )
    )

    // Agenda: 4 rows, 3 options each
    val AGENDA_GROUPS = listOf(0..2, 3..5, 6..8, 9..11)

    val AGENDA = CardTemplate(
        name = "Agenda",
        prefix = "AGN",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CORNERS,
        bubbleGroups = AGENDA_GROUPS,
        bubblePositions = listOf(
            PointF(308f, 260f), PointF(488f, 260f), PointF(668f, 260f),
            PointF(308f, 320f), PointF(488f, 320f), PointF(668f, 320f),
            PointF(308f, 380f), PointF(488f, 380f), PointF(668f, 380f),
            PointF(308f, 440f), PointF(488f, 440f), PointF(668f, 440f)
        )
    )

    fun fromPrefix(prefix: String): CardTemplate? =
        when {
            prefix.startsWith(CANDIDATE.prefix) -> CANDIDATE
            prefix.startsWith(AGENDA.prefix) -> AGENDA
            else -> null
        }
}
