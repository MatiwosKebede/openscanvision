package org.openscanvision.core.internal.omr

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
    const val REF_WIDTH = 850
    const val REF_HEIGHT = 540

    val SHARED_QR_CORNERS = listOf(
        PointF(592.5f, 52.5f),
        PointF(727.5f, 52.5f),
        PointF(727.5f, 187.5f),
        PointF(592.5f, 187.5f)
    )

    val SHARED_MARKER_CENTRES = listOf(
        PointF(37f, 37f),
        PointF(813f, 37f),
        PointF(813f, 503f),
        PointF(37f, 503f)
    )

    val ARUCO_TEMPLATE_CORNERS: Map<Int, List<PointF>> = mapOf(
        0 to listOf(PointF(12f, 12f), PointF(62f, 12f), PointF(62f, 62f), PointF(12f, 62f)),
        1 to listOf(PointF(788f, 12f), PointF(838f, 12f), PointF(838f, 62f), PointF(788f, 62f)),
        2 to listOf(PointF(788f, 478f), PointF(838f, 478f), PointF(838f, 528f), PointF(788f, 528f)),
        3 to listOf(PointF(12f, 478f), PointF(62f, 478f), PointF(62f, 528f), PointF(12f, 528f))
    )

    val CANDIDATE_GROUPS = listOf(0..11)
    val CANDIDATE = CardTemplate(
        name = "Candidate",
        prefix = "VX",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CENTRES,
        bubbleGroups = CANDIDATE_GROUPS,
        bubblePositions = listOf(
            PointF(90.2f, 264.2f),
            PointF(325.2f, 264.2f),
            PointF(577.2f, 264.2f),
            PointF(90.2f, 329.2f),
            PointF(328.2f, 329.2f),
            PointF(577.2f, 329.2f),
            PointF(90.2f, 393.2f),
            PointF(328.2f, 393.2f),
            PointF(577.2f, 393.2f),
            PointF(90.2f, 455.2f),
            PointF(328.2f, 455.2f),
            PointF(577.2f, 455.2f)
        )
    )

    val AGENDA_GROUPS = listOf(0..2, 3..5, 6..8, 9..11)
    val AGENDA = CardTemplate(
        name = "Agenda",
        prefix = "AGN",
        qrRefCorners = SHARED_QR_CORNERS,
        markerRefPositions = SHARED_MARKER_CENTRES,
        bubbleGroups = AGENDA_GROUPS,
        bubblePositions = listOf(
            PointF(303.2f, 264.2f), PointF(486.2f, 264.2f), PointF(668.2f, 264.2f),
            PointF(303.2f, 324.2f), PointF(486.2f, 324.2f), PointF(668.2f, 324.2f),
            PointF(303.2f, 384.2f), PointF(486.2f, 384.2f), PointF(668.2f, 384.2f),
            PointF(303.2f, 444.2f), PointF(486.2f, 444.2f), PointF(668.2f, 444.2f)
        )
    )

    fun fromPrefix(prefix: String): CardTemplate? =
        when {
            prefix.startsWith(CANDIDATE.prefix) -> CANDIDATE
            prefix.startsWith(AGENDA.prefix) -> AGENDA
            else -> null
        }
}
