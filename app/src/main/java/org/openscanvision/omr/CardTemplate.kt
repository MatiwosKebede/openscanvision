package org.openscanvision.omr

import android.graphics.PointF

/**
 * Card template definitions.
 * All coordinates are in 0.1 mm units (85 mm × 54 mm → 850 × 540).
 *
 * ArUco markers (5 mm × 5 mm) are placed 1.2 mm from the edges.
 * - Marker IDs: 0 = TL, 1 = TR, 2 = BR, 3 = BL.
 * - Marker corners are used for homography calculation.
 */
data class CardTemplate(
    val name: String,
    val prefix: String,
    val qrRefCorners: List<PointF>,
    val bubblePositions: List<PointF>,
    val markerRefPositions: List<PointF>? = null,
    val bubbleGroups: List<IntRange>? = null
)

object Templates {
    const val REF_WIDTH = 850
    const val REF_HEIGHT = 540

    // QR code wrapper (unchanged)
    val SHARED_QR_CORNERS = listOf(
        PointF(592.5f, 52.5f),   // TL
        PointF(727.5f, 52.5f),   // TR
        PointF(727.5f, 187.5f),  // BR
        PointF(592.5f, 187.5f)   // BL
    )

    // Marker centres (for centroid fallback)
    val SHARED_MARKER_CENTRES = listOf(
        PointF(37f, 37f),      // TL
        PointF(813f, 37f),     // TR
        PointF(813f, 503f),    // BR
        PointF(37f, 503f)      // BL
    )

    // ArUco marker corners (5mm squares, 1.2mm inset)
    val ARUCO_TEMPLATE_CORNERS: Map<Int, List<PointF>> = mapOf(
        0 to listOf(PointF(12f, 12f), PointF(62f, 12f), PointF(62f, 62f), PointF(12f, 62f)),
        1 to listOf(PointF(788f, 12f), PointF(838f, 12f), PointF(838f, 62f), PointF(788f, 62f)),
        2 to listOf(PointF(788f, 478f), PointF(838f, 478f), PointF(838f, 528f), PointF(788f, 528f)),
        3 to listOf(PointF(12f, 478f), PointF(62f, 478f), PointF(62f, 528f), PointF(12f, 528f))
    )

    // Candidate card: single race, 3 columns × 4 rows
    val CANDIDATE_GROUPS = listOf(0..11)
    val CANDIDATE = CardTemplate(
        name = "Candidate",
        prefix = "VX",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CENTRES,
        bubbleGroups = CANDIDATE_GROUPS,
        bubblePositions = listOf(
            PointF(64f, 247f), PointF(297f, 247f), PointF(530f, 247f),
            PointF(64f, 307f), PointF(297f, 307f), PointF(530f, 307f),
            PointF(64f, 367f), PointF(297f, 367f), PointF(530f, 367f),
            PointF(64f, 427f), PointF(297f, 427f), PointF(530f, 427f)
        )
    )

    // Agenda card: 4 rows, each with label column + 3 option columns
    val AGENDA_GROUPS = listOf(0..2, 3..5, 6..8, 9..11)
    val AGENDA = CardTemplate(
        name = "Agenda",
        prefix = "AGN",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CENTRES,
        bubbleGroups = AGENDA_GROUPS,
        bubblePositions = listOf(
            PointF(277f, 247f), PointF(457f, 247f), PointF(637f, 247f),
            PointF(277f, 307f), PointF(457f, 307f), PointF(637f, 307f),
            PointF(277f, 367f), PointF(457f, 367f), PointF(637f, 367f),
            PointF(277f, 427f), PointF(457f, 427f), PointF(637f, 427f)
        )
    )

    fun fromPrefix(prefix: String): CardTemplate? =
        when {
            prefix.startsWith(CANDIDATE.prefix) -> CANDIDATE
            prefix.startsWith(AGENDA.prefix) -> AGENDA
            else -> null
        }
}